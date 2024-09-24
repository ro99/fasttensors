use candle_core::cuda::cudarc::driver::sys as cudarc_sys;
use candle_core::cuda::cudarc::driver::DevicePtr;

use candle_core::DType;
use candle_core::Device;
use candle_core::Storage;
use candle_core::Tensor;
use io_uring::{opcode, types, IoUring};
use std::os::unix::io::AsRawFd;

use std::cell::UnsafeCell;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::ops::Deref;
use std::os::unix::fs::OpenOptionsExt;
use std::{
    ffi::OsStr,
    os::{
        raw::c_void,
        unix::io::RawFd,
    },
    path::Path,
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

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
    ptr: NonNull<u8>,
}

impl Page {
    pub fn new(ptr: NonNull<u8>) -> Self {
        Page {
            gpu_op_cell: CudaGpuOpCell::new(),
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

type PageCache = Cache<FileRange, Page>;

pub struct PinnedMemory {
    pinned_buffer: NonNull<u8>,
    pages: PageCache,
    ring: UnsafeCell<IoUring>, // New field
}

impl PinnedMemory {
    pub fn new() -> Result<Self, cudarc_sys::CUresult> {
        let size = PINNED_MEMORY;
        let alignment = MAX_BLOCK_SIZE;
        let total_size = size + alignment;

        let pinned_buffer = allocate_pinned_memory(total_size)?;

        let aligned = (pinned_buffer as usize + alignment - 1) & !(alignment - 1);
        let aligned_buffer = NonNull::new(aligned as *mut u8).unwrap();

        let pages = PageCache::new(MAX_PAGES);

        // Initialize pages
        for i in 0..MAX_PAGES {
            let fd = FileRange::new(-1 * i as i32, 0, 0);
            let page_ptr = unsafe { aligned_buffer.as_ptr().add(i * PAGE_SIZE) };
            let page = Page::new(NonNull::new(page_ptr).unwrap());
            pages.insert(fd, page);
        }

        let ring = IoUring::new(Q_DEPTH).expect("failed to create io_uring");

        Ok(PinnedMemory {
            pinned_buffer: NonNull::new(pinned_buffer).unwrap(),
            pages,
            ring: UnsafeCell::new(ring),
        })
    }

    pub fn get_cache_page(
        &self,
        file: &mut File,
        filesize: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Result<Arc<Page>, Box<dyn std::error::Error>> {
        let file_descriptor = file.as_raw_fd();
        let file_range = FileRange::new(file_descriptor, start_offset, end_offset);

        // Check if the page is already in cache
        if let Some(page) = self.pages.get(&file_range) {
            return Ok(page);
        }

        // If not in cache, we need to load it
        let (_, old_page) = self.pages.pop_lru().expect("failed to pop lru");
        let page_ptr = old_page.get_ptr().as_ptr();
        let chunk_size = PAGE_SIZE / Q_DEPTH as usize;
        let mut current_offset = start_offset;

        let ring = unsafe { &mut *self.ring.get() };
        let mut num_chunks = 0;

        // Prepare read operations
        for chunk_index in 0..Q_DEPTH {
            let next_offset = (current_offset + chunk_size).min(filesize);
            let read_size = next_offset - current_offset;

            let read_op = opcode::Read::new(
                types::Fd(file_descriptor),
                unsafe { page_ptr.add(chunk_index as usize * chunk_size) },
                read_size as _,
            )
            .offset(current_offset as _)
            .build();

            unsafe {
                ring.submission()
                    .push(&read_op)
                    .expect("failed to push read operation");
            }

            num_chunks += 1;
            if next_offset >= filesize {
                break;
            }
            current_offset = next_offset;
        }

        ring.submit_and_wait(num_chunks)?;
        ring.completion().next().expect("completion queue is empty");
        
        let page = Arc::new(Page::new(NonNull::new(page_ptr).unwrap()));

        // Insert the updated page into the cache
        self.pages.insert(file_range, page.clone());
        Ok(page)
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
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(path)?;

        let metadata = file.metadata()?;
        let filesize = metadata.len();

        Ok(Self {
            file,
            filesize,
        })
    }
    pub fn load(
        &mut self,
        tensor: &mut Tensor,
        device: &Device,
        offset: usize,
        length: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cuda_lib = get_cuda_lib();

        let pinned_memory = PinnedMemory::new().map_err(CudaError)?;

        let mut tensor_offset = 0;
        let mut file_b = offset / PAGE_SIZE * PAGE_SIZE;

        let (storage, _) = tensor.storage_mut_and_layout();

        let target = *match storage.deref() {
            Storage::Cuda(cuda_storage) => match storage.dtype() {
                DType::I32 => cuda_storage.as_cuda_slice::<i32>()?.device_ptr(),
                DType::I16 => cuda_storage.as_cuda_slice::<i16>()?.device_ptr(),
                DType::F16 => cuda_storage.as_cuda_slice::<half::f16>()?.device_ptr(),
                DType::BF16 => cuda_storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
                DType::F32 => cuda_storage.as_cuda_slice::<f32>()?.device_ptr(),
                DType::F64 => cuda_storage.as_cuda_slice::<f64>()?.device_ptr(),
                _ => unimplemented!("unsupported data type"),
            },
            _ => unreachable!("unexpected storage type"),
        } as *mut u8;

        let default_stream = match device {
            Device::Cuda(device) => device.cu_stream(),
            _ => return Err("Unsupported device".into()),
        };

        while tensor_offset < length {
            let file_a = file_b;
            file_b += PAGE_SIZE;

            let page = pinned_memory.get_cache_page(&mut self.file, self.filesize as usize, file_a, file_b)?;

            let left = offset.checked_sub(file_a).unwrap_or(0);
            let right = (offset + length - file_a).min(PAGE_SIZE);
            let copy_len = right - left;

            let src = unsafe { page.get_ptr().as_ptr().add(left) };
            let dst = unsafe { target.add(tensor_offset) };

            match device {
                Device::Cuda(_) => unsafe {
                    cuda_lib.cuMemcpyAsync(dst as u64, src as u64, copy_len, *default_stream).result()?;
                    page.gpu_op_cell.increment();
                    page.gpu_op_cell.add_callback_to_stream(cuda_lib, *default_stream)?;
                },
                _ => return Err("Unsupported device".into()),
            }
            tensor_offset += copy_len;
        }
        Ok(())
    }
}
