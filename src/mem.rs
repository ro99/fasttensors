use candle_core::cuda::cudarc::driver::sys as cudarc_sys;
use candle_core::cuda::cudarc::driver::DevicePtr;

use candle_core::DType;
use candle_core::Device;
use candle_core::Storage;
use candle_core::Tensor;
use std::io::SeekFrom;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::AtomicPtr;
use std::sync::{Arc, RwLock};
use tokio::fs::OpenOptions;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncSeekExt;

use std::io;
use std::ops::Deref;
use std::os::unix::fs::OpenOptionsExt;
use std::{
    ffi::OsStr,
    os::{raw::c_void, unix::io::RawFd},
    path::Path,
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};
use tokio::fs::File;

use crate::{
    cache::Cache,
    op::{allocate_pinned_memory, free_pinned_memory},
};

const Q_DEPTH: u32 = 8;
const MAX_PAGES: usize = 4;
const MAX_BLOCK_SIZE: usize = 128 * 1024; // 128 KB
const PAGE_SIZE: usize = 16 * 1024 * 1024; // 16 MB
const PINNED_MEMORY: usize = MAX_PAGES * PAGE_SIZE; // 64 MB
const O_DIRECT: i32 = 0x4000; // Linux

static CUDA_LIB: OnceLock<cudarc_sys::Lib> = OnceLock::new();
pub fn get_cuda_lib() -> &'static cudarc_sys::Lib {
    CUDA_LIB.get_or_init(|| {
        let cuda_path = OsStr::new("/usr/lib64/libcuda.so");
        unsafe { cudarc_sys::Lib::new(cuda_path).expect("Failed to load CUDA library") }
    })
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA error: {0:?}")]
struct CudaError(cudarc_sys::CUresult);

pub struct GpuOpCell {
    count: AtomicUsize,
}

impl GpuOpCell {
    pub fn new() -> Self {
        GpuOpCell {
            count: AtomicUsize::new(0),
        }
    }

    pub fn increment(&self) -> usize {
        self.count.fetch_add(1, Ordering::SeqCst)
    }

    pub fn decrement(&self) -> usize {
        self.count.fetch_sub(1, Ordering::SeqCst)
    }

    pub fn get(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    pub fn is_in_use(&self) -> bool {
        self.get() > 0
    }
}

pub struct CudaGpuOpCell {
    cell: Arc<GpuOpCell>,
}

impl CudaGpuOpCell {
    pub fn new() -> Self {
        CudaGpuOpCell {
            cell: Arc::new(GpuOpCell::new()),
        }
    }

    pub fn increment(&self) {
        self.cell.increment();
    }

    pub unsafe fn decrement_callback(
        &self,
        cuda_lib: &cudarc_sys::Lib,
        stream: cudarc_sys::CUstream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cell_clone = self.cell.clone();

        extern "C" fn callback(
            _stream: cudarc_sys::CUstream,
            _status: cudarc_sys::CUresult,
            user_data: *mut c_void,
        ) {
            let cell = unsafe { &*(user_data as *const GpuOpCell) };
            cell.decrement();
        }
        let user_data = Arc::into_raw(cell_clone) as *mut c_void;

        cuda_lib
            .cuStreamAddCallback(stream, Some(callback), user_data, 0)
            .result()?;

        Ok(())
    }
}

/*// This guard can be used to automatically decrement the counter when it goes out of scope
pub struct GpuOpGuard<'a> {
    cell: &'a CudaGpuOpCell,
}

impl<'a> GpuOpGuard<'a> {
    pub fn new(cell: &'a CudaGpuOpCell) -> Self {
        cell.increment();
        GpuOpGuard { cell }
    }
}

impl<'a> Drop for GpuOpGuard<'a> {
    fn drop(&mut self) {
        self.cell.cell.decrement();
    }
} */
// TODO: add a guard for the pinned memory

#[derive(Clone, Copy, Eq, Hash)]
pub struct FileRange {
    file_descriptor: RawFd,
    pub start: usize,
    pub end: usize,
}

impl FileRange {
    fn new(file_descriptor: RawFd, start: usize, end: usize) -> Self {
        FileRange {
            file_descriptor,
            start,
            end,
        }
    }
}

impl PartialEq for FileRange {
    fn eq(&self, other: &Self) -> bool {
        self.file_descriptor == other.file_descriptor
            && self.start == other.start
            && self.end == other.end
    }
}

impl Ord for FileRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.file_descriptor
            .cmp(&other.file_descriptor)
            .then(self.start.cmp(&other.start))
            .then(self.end.cmp(&other.end))
    }
}

impl PartialOrd for FileRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Page {
    gpu_op_cell: CudaGpuOpCell,
    file_range: FileRange,
    access: isize,
    ptr: NonNull<u8>,
}

impl Page {
    pub fn new(ptr: NonNull<u8>) -> Self {
        Page {
            gpu_op_cell: CudaGpuOpCell::new(),
            file_range: FileRange::new(-1, 0, 0),
            access: -1,
            ptr,
        }
    }
    pub fn is_in_use(&self) -> bool {
        self.gpu_op_cell.cell.is_in_use()
    }
    pub fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

pub struct PinnedMemory {
}

// Just to make sure we keep the pinned memory allocated
static PINNED_BUFFER: OnceLock<AtomicPtr<u8>> = OnceLock::new();
static ALIGNED_BUFFER: OnceLock<AtomicPtr<u8>> = OnceLock::new();

// Cache for the pages
static PAGES: OnceLock<Cache<FileRange, Page>> = OnceLock::new();

impl PinnedMemory {
    pub fn allocate() -> Result<(), cudarc_sys::CUresult> {
        if PINNED_BUFFER.get().is_none() {
            let size = PINNED_MEMORY;
            let alignment = MAX_BLOCK_SIZE;
            let total_size = size + alignment;

            let pinned_buffer = allocate_pinned_memory(total_size)?;
            PINNED_BUFFER.set(pinned_buffer.into()).expect("Failed to set PINNED_BUFFER");

            let aligned = (pinned_buffer as usize + alignment - 1) & !(alignment - 1);
            ALIGNED_BUFFER.set((aligned as *mut u8).into()).expect("Failed to set ALIGNED_BUFFER");

            let pages = Cache::new(MAX_PAGES);

            for i in 0..MAX_PAGES {
                let page_ptr = unsafe { (aligned as *mut u8).add(i * PAGE_SIZE) };
                let page = Page::new(NonNull::new(page_ptr).unwrap());
                pages.insert(FileRange::new(-1 * i as i32, 0, 0), page);
            }
            PAGES.set(pages).map_err(|_| cudarc_sys::CUresult::CUDA_ERROR_ILLEGAL_STATE)?;
        }
        Ok(())
    }

    pub async fn read_page(
        file: &mut File,
        file_range: FileRange,
        filesize: usize,
    ) -> Result<Arc<Page>, Box<dyn std::error::Error>> {
        if let Some(pages) = PAGES.get() {
            let (_, old_page) = pages.lru().unwrap();

            let read_len = std::cmp::min(file_range.end - file_range.start, filesize - file_range.start);

            file.seek(SeekFrom::Start(file_range.start as u64)).await?;
            let bytes_read = file.read(unsafe { std::slice::from_raw_parts_mut(old_page.ptr(), read_len) }).await?;
            if bytes_read == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Reached EOF after reading {} of {} bytes", bytes_read, read_len)
                )));
            }
            PAGES.get().unwrap().insert(file_range, old_page.clone());
            return Ok(old_page);
        }
        Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Page not found")))
    }

    pub async fn get_cache_page(
        file: &mut File,
        filesize: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Result<Arc<Page>, Box<dyn std::error::Error>> {
        let file_descriptor = file.as_raw_fd();
        let file_range = FileRange::new(file_descriptor, start_offset, end_offset);

        if let Some(page) = PAGES.get().unwrap().get(&file_range) {
            return Ok(page);
        }
        let page = PinnedMemory::read_page(file, file_range, filesize).await?;
        Ok(page)
    }
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        let pinned_buffer = PINNED_BUFFER.get().unwrap().load(std::sync::atomic::Ordering::SeqCst);
        if let Err(e) = free_pinned_memory(pinned_buffer) {
            eprintln!("Error freeing pinned memory: {:?}", e);
        }
    }
}

pub struct SafeTensorFile {
    file: File,
    filesize: u64,
}

impl SafeTensorFile {
    pub async fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(path)
            .await?;
        let metadata = file.metadata().await?;
        let filesize = metadata.len();

        Ok(Self { file, filesize })
    }

    pub async fn load(
        &mut self,
        tensor: &mut Tensor,
        device: &Device,
        offset: usize,
        length: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cuda_lib = get_cuda_lib();

        // Allocate pinned memory
        PinnedMemory::allocate().map_err(CudaError)?;

        let mut tensor_offset = 0;
        let mut file_b = offset / PAGE_SIZE * PAGE_SIZE;
        let mut file_a;

        // Get the raw data from the tensor
        let target = get_raw_tensor_ptr(tensor)?;

        // Get the default stream for the device  //TODO should we do a fork for each page?
        let default_stream = match device {
            Device::Cuda(device) => device.cuda_device().fork_default_stream()?,
            _ => return Err("Unsupported device".into()),
        };

        while tensor_offset < length {
            
            file_a = file_b;
            file_b += PAGE_SIZE;

            let page = PinnedMemory::get_cache_page(&mut self.file, self.filesize as usize, file_a, file_b).await?;

            let left = if file_a > offset { 0 } else { offset - file_a };
            let right = if (offset + length - file_a) > PAGE_SIZE { PAGE_SIZE } else { offset + length - file_a };
            let copy_len = right - left;

            let src = unsafe { page.ptr().add(left) };
            let dst = unsafe { target.add(tensor_offset) };

            match device {
                Device::Cuda(_) => unsafe {
                    cuda_lib.cuMemcpyAsync(dst as u64, src as u64, copy_len, default_stream.stream).result()?;
                    page.gpu_op_cell.increment();
                    page.gpu_op_cell.decrement_callback(cuda_lib, default_stream.stream)?;
                },
                _ => return Err("Unsupported device".into()),
            }
            tensor_offset += copy_len;
        }
        Ok(())
    }
}

fn get_raw_tensor_ptr(tensor: &mut Tensor) -> Result<*mut u8, Box<dyn std::error::Error>> {
    let (storage, _) = tensor.storage_mut_and_layout();

    let target = *match storage.deref() {
        Storage::Cuda(cuda_storage) => match storage.dtype() {
            DType::I32 => cuda_storage.as_cuda_slice::<i32>()?.device_ptr(),
            DType::I16 => cuda_storage.as_cuda_slice::<i16>()?.device_ptr(),
            DType::F16 => cuda_storage.as_cuda_slice::<half::f16>()?.device_ptr(),
            DType::BF16 => cuda_storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
            DType::F32 => cuda_storage.as_cuda_slice::<f32>()?.device_ptr(),
            DType::F64 => cuda_storage.as_cuda_slice::<f64>()?.device_ptr(),
            _ => return Err("unsupported data type".into()),
        },
        _ => return Err("unexpected storage type".into()),
    } as *mut u8;

    Ok(target)
}
