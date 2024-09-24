use candle_core::cuda::cudarc::driver::sys as cudarc_sys;
use candle_core::cuda::cudarc::driver::DevicePtr;

use candle_core::DType;
use candle_core::Device;
use candle_core::Storage;
use candle_core::Tensor;
use std::io::SeekFrom;
use std::os::unix::io::AsRawFd;
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

    pub unsafe fn add_callback_to_stream(
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
    start: usize,
    end: usize,
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

    pub fn get_ptr(&self) -> NonNull<u8> {
        self.ptr
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

pub struct PinnedMemory {
    pinned_buffer: NonNull<u8>,
    pages: Vec<Page>,
    serial: u64,
}

impl PinnedMemory {
    pub fn new() -> Result<Self, cudarc_sys::CUresult> {
        let size = PINNED_MEMORY;
        let alignment = MAX_BLOCK_SIZE;
        let total_size = size + alignment;

        let pinned_buffer = allocate_pinned_memory(total_size)?;

        let aligned = (pinned_buffer as usize + alignment - 1) & !(alignment - 1);
        let aligned_buffer = NonNull::new(aligned as *mut u8).unwrap();

        let mut pages = Vec::with_capacity(MAX_PAGES);

        // Initialize pages
        for i in 0..MAX_PAGES {
            let page_ptr = unsafe { aligned_buffer.as_ptr().add(i * PAGE_SIZE) };
            let page = Page::new(NonNull::new(page_ptr).unwrap());
            pages.push(page);
        }

        Ok(PinnedMemory {
            serial: 1,
            pinned_buffer: NonNull::new(pinned_buffer).unwrap(),
            pages,
        })
    }

    async fn read_page(
        &self,
        file: &mut File,
        page_ptr: *mut u8,
        offset: usize,
        filesize: usize,
    ) -> std::io::Result<usize> {
        let remaining_bytes = filesize.saturating_sub(offset);
        let read_len = remaining_bytes.min(PAGE_SIZE);

        let mut aligned_buffer = AlignedBuffer::new(read_len);

        file.seek(SeekFrom::Start(offset as u64)).await?;
        let bytes_read = file.read_exact(&mut aligned_buffer.buffer).await?;

        if bytes_read != read_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("Incomplete read: expected {}, got {}", read_len, bytes_read),
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(aligned_buffer.buffer.as_ptr(), page_ptr, read_len);
        }

        Ok(bytes_read)
    }

    fn find_page(&mut self, file_descriptor: RawFd, file_a: usize, file_b: usize) -> Option<usize> {
        let file_range = FileRange::new(file_descriptor, file_a, file_b);
        for (index, page) in self.pages.iter_mut().enumerate() {
            if page.file_range == file_range {
                page.access = self.serial as isize;
                self.serial += 1;
                return Some(index);
            }
        }
        None
    }

    fn evict_page(&mut self, file_descriptor: RawFd, file_a: usize, file_b: usize) -> usize {
        let mut oldest_i: Option<usize> = None;
        let mut oldest = std::isize::MAX;

        while oldest_i.is_none() {
            for (i, page) in self.pages.iter_mut().enumerate() {
                if page.is_in_use() {
                    continue;
                }
                if page.access < oldest {
                    oldest_i = Some(i);
                    oldest = page.access;
                }
            }
        }

        let p = oldest_i.expect("No page found to evict");

        self.pages[p].access = self.serial as isize;
        self.serial += 1;
        self.pages[p].file_range.start = file_a;
        self.pages[p].file_range.end = file_b;
        self.pages[p].file_range.file_descriptor = file_descriptor;

        p
    }

    pub async fn get_cache_page(
        &mut self,
        file: &mut File,
        filesize: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let file_descriptor = file.as_raw_fd();

        if let Some(page_index) = self.find_page(file_descriptor, start_offset, end_offset) {
            return Ok(page_index);
        }

        let page_index = self.evict_page(file_descriptor, start_offset, end_offset);
        let page = &self.pages[page_index];
        self.read_page(file, page.get_ptr().as_ptr(), start_offset, filesize).await?;
        Ok(page_index)

    }
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        if let Err(e) = free_pinned_memory(self.pinned_buffer.as_ptr()) {
            eprintln!("Error freeing pinned memory: {:?}", e);
        }
    }
}

// Aligned buffer for direct I/O
#[repr(align(4096))] // Typical alignment for direct I/O, adjust if needed
struct AlignedBuffer {
    buffer: Box<[u8]>,
}

impl AlignedBuffer {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0u8; size].into_boxed_slice(),
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

        let mut pinned_memory = PinnedMemory::new().map_err(CudaError)?;
        let mut tensor_offset = 0;
        let mut file_b = offset / PAGE_SIZE * PAGE_SIZE;

        let target = get_raw_tensor_ptr(tensor)?;
        let default_stream = match device {
            Device::Cuda(device) => device.cu_stream(),
            _ => return Err("Unsupported device".into()),
        };

        while tensor_offset < length {
            let file_a = file_b;
            file_b += PAGE_SIZE;

            let page_index = &pinned_memory.get_cache_page(&mut self.file, self.filesize as usize, file_a, file_b).await?;

            let page = &pinned_memory.pages[*page_index];

            let left = offset.checked_sub(file_a).unwrap_or(0);
            let right = (offset + length - file_a).min(PAGE_SIZE);
            let copy_len = right - left;

            let src = unsafe { page.get_ptr().as_ptr().add(left) };
            let dst = unsafe { target.add(tensor_offset) };

            match device {
                Device::Cuda(_) => unsafe {
                    cuda_lib
                        .cuMemcpyAsync(dst as u64, src as u64, copy_len, *default_stream)
                        .result()?;
                    page.gpu_op_cell.increment();
                    page.gpu_op_cell
                        .add_callback_to_stream(cuda_lib, *default_stream)?;
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
