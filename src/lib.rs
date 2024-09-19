use cache::{ContextCache, STFileCache, TensorCache};
use candle_core::cuda::cudarc::driver::sys::Lib;
use candle_core::{safetensors as sf, DType, Device, DeviceLocation, Error, Tensor};
use memmap2::Mmap;
use once_cell::sync::Lazy;
use safetensors::tensor::SafeTensorError;
use serde_json::Value;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use thiserror::Error;
//mod config;
mod cache;

const MAX_BLOCK_SIZE: usize = 128 * 1024;
const MAX_PAGES: usize = 4;
const PAGESIZE: usize = 16 * 1024 * 1024;
const Q_DEPTH: usize = 1;
const PINNED_MEMORY_SIZE: usize = MAX_PAGES * PAGESIZE;
const TENSOR_CACHE_CAPACITY: usize = 4;
const CONTEXT_CACHE_CAPACITY: usize = 100; //TODO: review this
const STFILE_CACHE_CAPACITY: usize = 100; //TODO: review this

pub(crate) static TENSOR_CACHE: Lazy<TensorCache> =
    // TENSOR_CACHE is equivalent to the python global_stfiles
    Lazy::new(|| TensorCache::new(TENSOR_CACHE_CAPACITY));
pub(crate) static CONTEXT_CACHE: Lazy<ContextCache> =
    // CONTEXT_CACHE is equivalent to the python global_cm
    Lazy::new(|| ContextCache::new(CONTEXT_CACHE_CAPACITY));
pub(crate) static STFILE_CACHE: Lazy<STFileCache> =
    Lazy::new(|| STFileCache::new(STFILE_CACHE_CAPACITY));

struct STPage {
    file_descriptor: i32,
    file_a: usize,
    file_b: usize,
    access: i64,
    locks: AtomicI32,
    ptr: *mut u8,
}

struct PinnedMemory {
    buffer: UnsafeCell<*mut u8>,
    aligned_buffer: UnsafeCell<*mut u8>,
    pages: [STPage; MAX_PAGES],
}

unsafe impl Sync for PinnedMemory {}
unsafe impl Send for PinnedMemory {}

static PINNED_MEMORY: Lazy<PinnedMemory> = Lazy::new(|| {
    let mut buffer: *mut u8 = std::ptr::null_mut();
    unsafe {
        Lib::cuMemAllocHost_v2(
            &mut buffer as *mut *mut u8 as *mut *mut std::ffi::c_void,
            PINNED_MEMORY_SIZE + MAX_BLOCK_SIZE,
        );
    }
    assert!(!buffer.is_null(), "Unable to allocate pinned memory");
    let aligned_buffer =
        unsafe { buffer.add(MAX_BLOCK_SIZE - 1) as usize & !(MAX_BLOCK_SIZE - 1) } as *mut u8;

    let pages = std::array::from_fn(|i| STPage {
        file_descriptor: -1,
        file_a: 0,
        file_b: 0,
        access: -1,
        locks: AtomicI32::new(0),
        ptr: unsafe { aligned_buffer.add(i * PAGESIZE) },
    });

    PinnedMemory {
        buffer: UnsafeCell::new(buffer),
        aligned_buffer: UnsafeCell::new(aligned_buffer),
        pages,
    }
});
pub struct STFile {
    filename: String,
    header: serde_json::Value,
    header_size: usize,
    metadata: Option<serde_json::Value>,
    handle: std::fs::File,
    fast: bool,
    tensor_remap: Option<HashMap<String, String>>,
}

impl STFile {
    pub fn init<P: AsRef<Path>>(
        filename: P,
        fast: bool,
        keymap: Option<Vec<(String, String)>>,
    ) -> Result<(), Error> {
        let filename = filename.as_ref().to_str().unwrap().to_string();
        let mut st_file = Self {
            filename: filename.clone(),
            header: serde_json::Value::Null,
            header_size: 0,
            metadata: None,
            handle: std::fs::File::open(&filename)?,
            fast,
            tensor_remap: None,
        };

        st_file.read_dict()?;

        if let Some(keymap) = keymap {
            let mut tensor_remap = HashMap::new();
            let mut new_header = serde_json::Map::new();

            for (key, value) in st_file.header.as_object().unwrap() {
                let mut new_key = key.clone();
                for (from, to) in &keymap {
                    if from.starts_with('$') && new_key.starts_with(&from[1..]) {
                        new_key = format!("${}", new_key).replace(from, to);
                    } else {
                        new_key = new_key.replace(from, to);
                    }
                }
                new_header.insert(new_key.clone(), value.clone());
                tensor_remap.insert(new_key, key.clone());
            }

            st_file.header = Value::Object(new_header);
            st_file.tensor_remap = Some(tensor_remap);
        }

        if st_file.fast {
            //st_file.handle = ext_c.safetensors_open
            todo!()
        }

        STFILE_CACHE.insert(st_file.filename.clone(), st_file);

        Ok(())
    }

    fn read_dict(&mut self) -> Result<(), STFileError> {
        let file = File::open(&self.filename)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read header size (first 8 bytes)
        let header_size = u64::from_le_bytes(mmap[..8].try_into()?) as usize;

        // Parse header JSON
        let header_end = 8 + header_size;
        let header: Value = serde_json::from_slice(&mmap[8..header_end])?;

        // Extract metadata if present
        if let Some(obj) = header.as_object() {
            if let Some(metadata) = obj.get("__metadata__") {
                self.metadata = Some(metadata.clone());
            }
        }

        // Store header without metadata
        self.header = if let Value::Object(mut obj) = header {
            obj.remove("__metadata__");
            Value::Object(obj)
        } else {
            header
        };

        self.header_size = header_end;

        Ok(())
    }

    pub fn open<P: AsRef<Path>>(
        filename: P,
        fast: bool,
        keymap: Option<Vec<(String, String)>>,
    ) -> Result<Arc<Self>, Error> {
        let key = filename.as_ref().to_str().unwrap().to_string();
        if let Some(st_file) = STFILE_CACHE.get(&key) {
            return Ok(st_file.clone());
        }
        Self::init(filename, fast, keymap)?;
        Ok(STFILE_CACHE.get(&key).unwrap().clone())
    }

    pub fn close(&mut self) -> Result<(), Error> {
        // Implementation
        todo!()
    }

    pub fn get_dict(&self) -> &serde_json::Value {
        &self.header
    }

    pub fn get_metadata(&self) -> Option<&serde_json::Value> {
        self.metadata.as_ref()
    }

    pub fn measure(&self, key: &str) -> Result<usize, Error> {
        let v = self
            .header
            .get(key)
            .ok_or_else(|| Error::SafeTensor(SafeTensorError::TensorNotFound(key.to_string())))?;

        let data_offsets = v["data_offsets"]
            .as_array()
            .ok_or_else(|| Error::Msg(format!("Invalid data_offsets for key: {}", key)))?;

        if data_offsets.len() != 2 {
            return Err(Error::Msg(format!(
                "Expected 2 data_offsets, found {} for key: {}",
                data_offsets.len(),
                key
            )));
        }

        let start = data_offsets[0]
            .as_u64()
            .ok_or_else(|| Error::Msg(format!("Invalid start offset for key: {}", key)))?
            as usize;
        let end = data_offsets[1]
            .as_u64()
            .ok_or_else(|| Error::Msg(format!("Invalid end offset for key: {}", key)))?
            as usize;

        Ok(end - start)
    }

    pub fn get_tensor(
        &self,
        key: &String,
        device: &Device,
        not_fast: bool,
        out_dtype: Option<DType>,
    ) -> Result<Tensor, Error> {
        let _ = device.synchronize(); //TODO review this

        let device_id = match device.location() {
            DeviceLocation::Cpu => "cpu".to_string(),
            DeviceLocation::Cuda { gpu_id } => format!("cuda:{}", gpu_id),
            DeviceLocation::Metal { gpu_id } => format!("metal:{}", gpu_id),
        };

        let key = if self.tensor_remap.is_some() && (not_fast || !self.fast) {
            self.tensor_remap.as_ref().unwrap().get(key).unwrap_or(key)
        } else {
            key
        };

        let cache_key = format!("{}::{}::{}", self.filename, key, device_id);

        if let Some(cached_tensor) = TENSOR_CACHE.get(&cache_key) {
            return Ok(cached_tensor.as_ref().clone());
        }


        let tensor = if !self.fast || not_fast {
            self.load_tensor_slow(key, device)?
        } else {
            self.load_tensor_fast(key, device)?
        };

        // Add to cache if caching is enabled
        //if cached {
        //    GLOBAL_TENSOR_CACHE.insert(cache_key, tensor.clone());
        //}

        todo!()
    }

    fn load_tensor_slow(&self, key: &str, device: &Device) -> Result<Tensor, Error> {
        let device_id = match device.location() {
            DeviceLocation::Cpu => "cpu".to_string(),
            DeviceLocation::Cuda { gpu_id } => format!("cuda:{}", gpu_id),
            DeviceLocation::Metal { gpu_id } => format!("metal:{}", gpu_id),
        };
        let cache_key = format!("{}::{}", self.filename, device_id);

        if let Some(context) = CONTEXT_CACHE.get(&cache_key) {
            return context
                .get(key)
                .map(|tensor| tensor.clone())
                .ok_or_else(|| {
                    Error::SafeTensor(SafeTensorError::TensorNotFound(key.to_string()))
                });
        }
        let context = sf::load(&self.filename, device)
            .map_err(|_| Error::SafeTensor(SafeTensorError::TensorNotFound(key.to_string())))?;
        CONTEXT_CACHE.insert(cache_key, context.clone());

        let tensor = context
            .get(key)
            .ok_or_else(|| Error::SafeTensor(SafeTensorError::TensorNotFound(key.to_string())))?;

        Ok(tensor.clone())
    }

    fn load_tensor_fast(&self, key: &str, device: &Device) -> Result<Tensor, Error> {
        let cuda_device = match device.location() {
            DeviceLocation::Cuda { gpu_id } => CudaDevice::new(gpu_id)?,
            _ => {
                return Err(Error::Msg(
                    "Fast tensor loading is only supported for CUDA devices".to_string(),
                ))
            }
        };

        let tensor_info = self
            .header
            .get(key)
            .ok_or_else(|| Error::SafeTensor(SafeTensorError::TensorNotFound(key.to_string())))?;

        let dtype = convert_dtype(&tensor_info["dtype"].as_str().unwrap())?;
        let shape: Vec<usize> = tensor_info["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let data_offsets = tensor_info["data_offsets"].as_array().unwrap();
        let offset = data_offsets[0].as_u64().unwrap() as usize + self.header_size;
        let length =
            (data_offsets[1].as_u64().unwrap() - data_offsets[0].as_u64().unwrap()) as usize;

        // Ensure the tensor shape matches the storage size
        let expected_size = shape.iter().product::<usize>() * dtype.size_in_bytes();
        if expected_size != length {
            return Err(Error::Msg(format!(
                "Tensor shape doesn't match storage size for key: {}",
                key
            )));
        }

        // Allocate device memory
        let mut device_buffer: CudaSlice<u8> = unsafe { cuda_device.alloc(length)? };

        // Load data using pinned memory and async copy
        let mut tensor_offset = 0;
        let mut file_b = offset / PAGESIZE * PAGESIZE;

        while tensor_offset < length {
            let file_a = file_b;
            file_b += PAGESIZE;

            let page = self.get_cache_page(file_a, file_b)?;
            let left = (offset - file_a).max(0) as usize;
            let right = (offset + length - file_a).min(PAGESIZE) as usize;
            let copy_len = right - left;

            let src = unsafe { page.ptr.add(left) };
            let dst = unsafe { device_buffer.as_mut_ptr().add(tensor_offset) };

            page.locks.fetch_add(1, Ordering::SeqCst);
            unsafe {
                cuda_device.memcpy_async(
                    dst,
                    src,
                    copy_len,
                    MemcpyKind::HostToDevice,
                    None, // Use default stream
                )?;
            }
            cuda_device.add_callback(None, move |_| {
                page.locks.fetch_sub(1, Ordering::SeqCst);
            })?;

            tensor_offset += copy_len;
        }

        // Create a Tensor from the device buffer
        let tensor = Tensor::from_cuda_slice(device_buffer, &shape, dtype)?;

        Ok(tensor)
    }

    fn get_cache_page(&self, file_a: usize, file_b: usize) -> Result<&STPage, Error> {
        let mut oldest_i = 0;
        let mut oldest = i64::MAX;

        // Find existing page in cache or the oldest page to evict
        for (i, page) in PINNED_MEMORY.pages.iter().enumerate() {
            if page.file_descriptor == self.file_descriptor
                && page.file_a == file_a
                && page.file_b == file_b
            {
                return Ok(page);
            }
            if page.locks.load(Ordering::SeqCst) == 0 && page.access < oldest {
                oldest_i = i;
                oldest = page.access;
            }
        }

        // Load new page
        let page = &mut PINNED_MEMORY.pages[oldest_i];
        page.file_a = file_a;
        page.file_b = file_b;
        page.file_descriptor = self.file_descriptor;

        // Use Linux AIO to read the page
        let mut aiocb = nix::sys::aio::AioCb::from_slice(
            self.file_descriptor,
            file_a as i64,
            unsafe { std::slice::from_raw_parts_mut(page.ptr, PAGESIZE) },
            0,
            nix::sys::aio::LioOpcode::LIO_READ,
        );

        nix::sys::aio::aio_read(&mut aiocb)?;
        nix::sys::aio::aio_suspend(&[&aiocb], None)?;

        let result = nix::sys::aio::aio_return(&aiocb)?;
        if result as usize != PAGESIZE {
            return Err(Error::Msg("Async read error".to_string()));
        }

        Ok(page)
    }
}

pub fn cleanup_stfiles() {
    // Implementation
    todo!()
}

pub fn convert_dtype(dt: &str) -> Result<DType, Error> {
    // Implementation here
    todo!()
}

#[derive(Error, Debug)]
pub enum STFileError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Slice conversion error: {0}")]
    Conversion(#[from] std::array::TryFromSliceError),
}

impl From<STFileError> for Error {
    fn from(error: STFileError) -> Self {
        match error {
            STFileError::Io(e) => Error::Io(e),
            STFileError::Json(e) => {
                Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
            STFileError::InvalidData(s) => {
                Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, s))
            }
            STFileError::Conversion(e) => {
                Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
        }
    }
}
