use candle_core::cuda::cudarc::driver::{sys as cudarc_sys, DevicePtr, sys::CUresult};
use candle_core::{DType, Device, Storage, Tensor};
use tokio::io::{SeekFrom, AsyncReadExt, AsyncSeekExt};
use tokio::fs::{File, OpenOptions};
use tokio::time::{sleep, Instant};

use std::os::{unix::io::RawFd, raw::c_void, unix::io::AsRawFd};
use std::sync::atomic::AtomicI32;
use std::sync::{Arc, OnceLock, atomic::{Ordering, AtomicPtr}};
use std::time::Duration;
use std::{ffi::OsStr, path::Path, ptr::NonNull};
use std::ops::Deref;

use crate::cache::Cache;

#[cfg(target_os = "linux")]
use libc::O_DIRECT;

#[cfg(not(target_os = "linux"))]
const O_DIRECT: i32 = 0;

const MAX_PAGES: usize = 8;
const MAX_BLOCK_SIZE: usize = 128 * 1024; // 128 KB
const PAGE_SIZE: usize = 32 * 1024 * 1024; // 32 MB
const PINNED_MEMORY: usize = MAX_PAGES * PAGE_SIZE; // 128 MB

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
    count: AtomicI32,
}

impl GpuOpCell {
    pub fn new() -> Self {
        GpuOpCell {
            count: AtomicI32::new(0),
        }
    }

    pub fn increment(&self) -> i32 {
        self.count.fetch_add(1, Ordering::SeqCst)
    }

    pub fn decrement(&self) -> i32 {
        self.count.fetch_sub(1, Ordering::SeqCst)
    }

    pub fn get(&self) -> i32 {
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
    pub fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

// Just to make sure we keep the pinned memory allocated
static PINNED_BUFFER: OnceLock<AtomicPtr<u8>> = OnceLock::new();
static ALIGNED_BUFFER: OnceLock<AtomicPtr<u8>> = OnceLock::new();

// Cache for the pages
static PAGES: OnceLock<Cache<FileRange, Page>> = OnceLock::new();

pub struct SafeTensorFile {
    file_descriptor: File,
    filesize: u64,
}

// Ensure SafeTensorFile is Send + Sync
unsafe impl Send for SafeTensorFile {}
unsafe impl Sync for SafeTensorFile {}

impl SafeTensorFile {
    pub async fn new<P: AsRef<Path>>(path: P) -> tokio::io::Result<Self> {
        let file_descriptor = OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(path)
            .await?;
        let metadata = file_descriptor.metadata().await?;
        let filesize = metadata.len();

        Ok(Self {
            file_descriptor,
            filesize,
        })
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
        allocate().map_err(CudaError)?;

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
            file_b = file_a + PAGE_SIZE;

            let page = get_cache_page(&mut self.file_descriptor, self.filesize as usize, file_a, file_b).await?;

            let left = if file_a > offset { 0 } else { offset - file_a };
            let right = if file_a > offset + length { 0 } else {
                let x = offset + length - file_a;
                if x > PAGE_SIZE { PAGE_SIZE } else { x }
            };

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


async fn read_page(
    file: &mut File,
    file_range: FileRange,
    filesize: usize,
) -> Result<Arc<Page>, Box<dyn std::error::Error>> {
    if let Some(pages) = PAGES.get() {
        const MAX_WAIT_TIME: Duration = Duration::from_secs(5);
        const WAIT_INTERVAL: Duration = Duration::from_millis(10);
        let start_time = Instant::now();

        loop {
            // Find all pages that are not in use
            let available_pages: Vec<_> = pages.iter()
                .into_iter()  // Convert Vec back to an iterator
                .filter(|(_, page)| !page.is_in_use())
                .collect();

            if !available_pages.is_empty() {
                // Find the least recently used page among the available ones
                let (evict_key, evict_page) = available_pages
                    .into_iter()
                    .min_by_key(|(k, _)| pages.get_last_use(k))
                    .unwrap();

                // Remove the page from its current position
                pages.remove(&evict_key);

                let read_len = std::cmp::min(
                    PAGE_SIZE,
                    filesize - file_range.start,
                );

                let mut buffer = unsafe { std::slice::from_raw_parts_mut(evict_page.ptr(), read_len) };
                file.seek(SeekFrom::Start(file_range.start as u64)).await?;
                let bytes_read = file.read_exact(&mut buffer).await?;

                if bytes_read == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "Reached EOF after reading {} of {} bytes",
                            bytes_read, read_len
                        ),
                    )));
                }

                pages.insert(file_range, evict_page.clone());
                return Ok(evict_page);
            }

            if start_time.elapsed() > MAX_WAIT_TIME {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Timed out waiting for an available page",
                )));
            }

            sleep(WAIT_INTERVAL).await;
        }
    }
    Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        "Page not found",
    )))
}

async fn get_cache_page(
    file_descriptor: &mut File,
    filesize: usize,
    start_offset: usize,
    end_offset: usize,
) -> Result<Arc<Page>, Box<dyn std::error::Error>> {
    let file_descriptor_fd = file_descriptor.as_raw_fd();
    let file_range = FileRange::new(file_descriptor_fd, start_offset, end_offset);

    if let Some(page) = PAGES.get().unwrap().get(&file_range) {
        return Ok(page);
    }
    let page = read_page(file_descriptor, file_range, filesize).await?;
    Ok(page)
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

pub fn allocate() -> Result<(), cudarc_sys::CUresult> {
    if PINNED_BUFFER.get().is_none() {
        let size = PINNED_MEMORY;
        let alignment = align_of::<usize>().max(MAX_BLOCK_SIZE);
        let total_size = size + alignment;

        let pinned_buffer = allocate_pinned_memory(total_size)?;
        PINNED_BUFFER
            .set(AtomicPtr::new(pinned_buffer))
            .expect("Failed to set PINNED_BUFFER");

        let aligned_offset = pinned_buffer.align_offset(alignment);
        let aligned = unsafe { pinned_buffer.add(aligned_offset) };
        ALIGNED_BUFFER
            .set(AtomicPtr::new(aligned))
            .expect("Failed to set ALIGNED_BUFFER");

        let pages = Cache::new(MAX_PAGES);

        for i in 0..MAX_PAGES {
            let page_ptr = unsafe { (aligned as *mut u8).add(i * PAGE_SIZE) };
            let page = Page::new(NonNull::new(page_ptr).unwrap());
            pages.insert(FileRange::new(-1, 0, 0), page);
        }
        PAGES
            .set(pages)
            .map_err(|_| cudarc_sys::CUresult::CUDA_ERROR_ILLEGAL_STATE)?;
    }
    Ok(())
}

pub fn allocate_pinned_memory(size: usize) -> Result<*mut u8, CUresult> {
    let cuda_lib = get_cuda_lib();
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        let result = cuda_lib.cuMemAllocHost_v2(&mut ptr as *mut _, size);
        if result != CUresult::CUDA_SUCCESS {
            return Err(result);
        }
    }
    Ok(ptr as *mut u8)
}

pub fn free_pinned_memory() -> Result<(), CUresult> {
    let cuda_lib = get_cuda_lib();
    
    if let Some(pinned_buffer) = PINNED_BUFFER.get() {
        let ptr = pinned_buffer.swap(std::ptr::null_mut(), Ordering::SeqCst);
        if !ptr.is_null() {
            unsafe {
                let result = cuda_lib.cuMemFreeHost(ptr as *mut std::ffi::c_void);
                if result != CUresult::CUDA_SUCCESS {
                    return Err(result);
                }
            }
        }
    }

    if let Some(aligned_buffer) = ALIGNED_BUFFER.get() {
        aligned_buffer.store(std::ptr::null_mut(), Ordering::SeqCst);
    }

    PAGES.get().unwrap().clear();

    Ok(())
}