use candle_core::cuda::cudarc::driver::{
    sys::{CUresult, CUstream, Lib},
    CudaStream,
};
use io_uring::{opcode, types, IoUring};
use std::os::raw::c_void;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::{ffi::OsStr, mem::MaybeUninit};
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

static CUDA_LIB: OnceLock<Lib> = OnceLock::new();
pub fn get_cuda_lib() -> &'static Lib {
    CUDA_LIB.get_or_init(|| {
        let cuda_path = OsStr::new("/usr/local/cuda/lib64/libcuda.so");
        unsafe { Lib::new(cuda_path).expect("Failed to load CUDA library") }
    })
}

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
        cuda_lib: &Lib,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cell_clone = self.cell.clone();

        extern "C" fn callback(_stream: CUstream, _status: CUresult, user_data: *mut c_void) {
            let cell = unsafe { &*(user_data as *const GpuOpCell) };
            cell.decrement();
        }
        let user_data = Arc::into_raw(cell_clone) as *mut c_void;

        cuda_lib
            .cuStreamAddCallback(stream.stream, Some(callback), user_data, 0)
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
    file_range: FileRange,
    gpu_op_cell: CudaGpuOpCell,
    ptr: NonNull<u8>,
}

impl Page {
    pub fn new(fd: FileRange, ptr: NonNull<u8>) -> Self {
        Page {
            file_range: fd,
            gpu_op_cell: CudaGpuOpCell::new(),
            ptr,
        }
    }

    pub fn is_fd_assigned(&self) -> bool {
        self.file_range.file_descriptor > 0
    }

    pub fn is_in_use(&self) -> bool {
        self.gpu_op_cell.cell.is_in_use()
    }

    pub fn get_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    pub fn get_file_range(&self) -> &FileRange {
        &self.file_range
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

type PageCache = Cache<FileRange, Page>;

pub struct PinnedMemory {
    pinned_buffer: NonNull<u8>,
    aligned_buffer: NonNull<u8>,
    pages: PageCache,
    size: usize,
}

impl PinnedMemory {
    pub fn new() -> Result<Self, CUresult> {
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
            let page = Page::new(fd, NonNull::new(page_ptr).unwrap());
            pages.insert(fd, page);
        }

        Ok(PinnedMemory {
            pinned_buffer: NonNull::new(pinned_buffer).unwrap(),
            aligned_buffer,
            pages,
            size,
        })
    }

    pub async fn get_cache_page(
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
        let mut ring = IoUring::new(Q_DEPTH).expect("failed to create io_uring");
        let (_, old_page) = self.pages.pop_lru().expect("failed to pop lru");
        let page_ptr = old_page.get_ptr().as_ptr();
        let chunk_size = PAGE_SIZE / Q_DEPTH as usize;
        let mut current_offset = start_offset;

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

            if next_offset >= filesize {
                break;
            }
            current_offset = next_offset;
        }

        // Submit and wait for completion
        ring.submit_and_wait(Q_DEPTH as usize)?;

        // Process completions
        ring.completion().for_each(|cqe| {
            if cqe.result() < 0 {
                eprintln!(
                    "Read error: {}",
                    std::io::Error::from_raw_os_error(-cqe.result())
                );
            }
        });

        let page = Arc::new(Page::new(file_range, NonNull::new(page_ptr).unwrap()));

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
