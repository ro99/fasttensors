use cache::Cache;
use candle_core::{safetensors as sf, DType, Device, DeviceLocation, Tensor};
use mem::SafeTensorFile;
use memmap2::Mmap;
use safetensors::tensor::SafeTensorError;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};



//mod config;
mod cache;
mod mem;
mod op;

const TENSOR_CACHE_CAPACITY: usize = 4;
const CONTEXT_CACHE_CAPACITY: usize = 100; //TODO: review this
const STFILE_CACHE_CAPACITY: usize = 100; //TODO: review this

pub type TensorCache = Cache<String, Tensor>;
pub type ContextCache = Cache<String, HashMap<String, Tensor>>;
pub type STFileCache = Cache<String, FastTensorFile>;

#[derive(thiserror::Error, Debug)]
pub enum FastTensorsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] SafeTensorError),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Slice conversion error: {0}")]
    Conversion(#[from] std::array::TryFromSliceError),
    #[error("Unknown dtype: {0}")]
    UnknownDtype(String),
}

impl From<FastTensorsError> for candle_core::Error {
    fn from(error: FastTensorsError) -> Self {
        match error {
            FastTensorsError::Io(e) => candle_core::Error::Io(e),
            FastTensorsError::Json(e) => {
                candle_core::Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
            FastTensorsError::InvalidData(s) => {
                candle_core::Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, s))
            }
            FastTensorsError::Conversion(e) => {
                candle_core::Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
            FastTensorsError::UnknownDtype(s) => {
                candle_core::Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, s))
            }
            FastTensorsError::SafeTensors(e) => todo!(),
            FastTensorsError::Candle(e) => todo!(),
        }
    }
}

pub(crate) static TENSOR_CACHE: OnceLock<TensorCache> = OnceLock::new();
pub(crate) static CONTEXT_CACHE: OnceLock<ContextCache> = OnceLock::new();
pub(crate) static STFILE_CACHE: OnceLock<STFileCache> = OnceLock::new();

// Initialize the caches
pub fn init_caches() {
    TENSOR_CACHE.get_or_init(|| TensorCache::new(TENSOR_CACHE_CAPACITY));
    CONTEXT_CACHE.get_or_init(|| ContextCache::new(CONTEXT_CACHE_CAPACITY));
    STFILE_CACHE.get_or_init(|| STFileCache::new(STFILE_CACHE_CAPACITY));
}

pub struct FastTensorFile {
    filename: String,
    header: serde_json::Value,
    header_size: usize,
    metadata: Option<serde_json::Value>,
    tensor_remap: Option<HashMap<String, String>>,
    fast: bool,
    file_handle: Mutex<SafeTensorFile>,
}

impl FastTensorFile {
    pub async fn new<P: AsRef<Path>>(
        filename: P,
        fast: bool,
        keymap: Option<Vec<(String, String)>>,
    ) -> Result<Arc<Self>, FastTensorsError> {

        // Check if the file is already in the cache
        let filename_str = filename.as_ref().to_string_lossy().into_owned();
        if let Some(cached_file) = STFILE_CACHE.get().unwrap().get(&filename_str) {
            return Ok(cached_file.clone());
        }

        // Read from disk and map into memory
        let file = File::open(&filename)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read header size (first 8 bytes)
        let header_size = u64::from_le_bytes(mmap[..8].try_into().map_err(|e| {
            FastTensorsError::InvalidData(format!("Failed to convert slice: {}", e))
        })?) as usize;

        // Parse header JSON
        let mut header: Value = serde_json::from_slice(&mmap[8..8 + header_size])
            .map_err(|e| FastTensorsError::InvalidData(format!("JSON parsing error: {}", e)))?;

        // Extract metadata if present
        let metadata = header.get("__metadata__").cloned();
        if metadata.is_some() {
            header.as_object_mut().unwrap().remove("__metadata__");
        }

        let tensor_remap = keymap.map(|keymap| {
            let mut remap = HashMap::new();
            let mut new_header = serde_json::Map::new();

            for (key, value) in header.as_object().unwrap() {
                let mut new_key = key.clone();
                for (from, to) in &keymap {
                    if from.starts_with('$') && new_key.starts_with(&from[1..]) {
                        new_key = format!("${}", new_key).replace(from, to);
                    } else {
                        new_key = new_key.replace(from, to);
                    }
                }
                new_header.insert(new_key.clone(), value.clone());
                remap.insert(new_key, key.clone());
            }
            header = Value::Object(new_header);
            remap
        });

        let file = SafeTensorFile::new(filename_str.clone()).await?;

        let stfile = Arc::new(Self {
            filename: filename_str.clone(),
            header,
            header_size,
            metadata,
            tensor_remap,
            fast,
            file_handle: Mutex::new(file),
        });


        STFILE_CACHE.get().unwrap().insert(filename_str.clone(), stfile.clone());
        Ok(stfile)
    }

    pub fn close(&mut self) -> Result<(), FastTensorsError> {
        if self.fast {}

        Ok(())
    }

    pub fn get_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.header.as_object().unwrap().keys().cloned().collect();
        keys.sort_by(|a, b| {
            let a_offset = self.header[a]["data_offsets"][0].as_u64().unwrap();
            let b_offset = self.header[b]["data_offsets"][0].as_u64().unwrap();
            a_offset.cmp(&b_offset)
        });
        keys
    }
    pub fn get_dict(&self) -> &serde_json::Value {
        &self.header
    }

    pub fn get_metadata(&self) -> Option<&serde_json::Value> {
        self.metadata.as_ref()
    }

    pub fn measure(&self, key: &str) -> Result<usize, FastTensorsError> {
        let v = self.header.get(key).ok_or_else(|| {
            FastTensorsError::SafeTensors(SafeTensorError::TensorNotFound(key.to_string()))
        })?;

        let data_offsets = v["data_offsets"].as_array().ok_or_else(|| {
            FastTensorsError::InvalidData(format!("Invalid data_offsets for key: {}", key))
        })?;

        if data_offsets.len() != 2 {
            return Err(FastTensorsError::InvalidData(format!(
                "Expected 2 data_offsets, found {} for key: {}",
                data_offsets.len(),
                key
            )));
        }

        let start = data_offsets[0].as_u64().ok_or_else(|| {
            FastTensorsError::InvalidData(format!("Invalid start offset for key: {}", key))
        })? as usize;
        let end = data_offsets[1].as_u64().ok_or_else(|| {
            FastTensorsError::InvalidData(format!("Invalid end offset for key: {}", key))
        })? as usize;

        Ok(end - start)
    }

    pub async fn get_tensor(
        &self,
        key: &String,
        device: &Device,
        not_fast: bool,
        out_dtype: Option<DType>,
    ) -> Result<Arc<Tensor>, FastTensorsError> {
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

        if let Some(cached_tensor) = TENSOR_CACHE.get().unwrap().get(&cache_key) {
            return Ok(cached_tensor.clone());
        }

        let mut tensor = if not_fast {
            self.load_tensor_slow(key, device)?
        } else {
            self.load_tensor_fast(key, device).await?
        };

        if let Some(out_dtype) = out_dtype {
            tensor = tensor.to_dtype(out_dtype)?;
        }

        let tensor = Arc::new(tensor);

        TENSOR_CACHE
            .get()
            .unwrap()
            .insert(cache_key, tensor.clone());

        Ok(tensor)
    }

    fn load_tensor_slow(
        &self,
        key: &str,
        device: &Device,
    ) -> Result<Tensor, FastTensorsError> { 
        let device_id = match device.location() {
            DeviceLocation::Cpu => "cpu".to_string(),
            DeviceLocation::Cuda { gpu_id } => format!("cuda:{}", gpu_id),
            DeviceLocation::Metal { gpu_id } => format!("metal:{}", gpu_id),
        };
        let cache_key = format!("{}::{}", self.filename, device_id);

        if let Some(context) = CONTEXT_CACHE.get().unwrap().get(&cache_key) {
            return context
                .get(key)
                .map(|tensor| tensor.clone())
                .ok_or_else(|| {
                    FastTensorsError::SafeTensors(SafeTensorError::TensorNotFound(key.to_string()))
                });
        }

        let context = sf::load(&self.filename, device).map_err(|e| {
            println!("Error loading context: {}", e);
            FastTensorsError::SafeTensors(SafeTensorError::TensorNotFound(key.to_string()))
        })?;

        CONTEXT_CACHE.get().unwrap().insert(cache_key, context.clone());

        let tensor = context.get(key).ok_or_else(|| {
            FastTensorsError::SafeTensors(SafeTensorError::TensorNotFound(key.to_string()))
        })?;

        Ok(tensor.clone())
    }

    async fn load_tensor_fast(
        &self,
        key: &str,
        device: &Device,
    ) -> Result<Tensor, FastTensorsError> {
        let header_info = self.header.get(key).ok_or_else(|| {
            FastTensorsError::SafeTensors(SafeTensorError::TensorNotFound(key.to_string()))
        })?;
        let dtype = convert_dtype(&header_info["dtype"].as_str().unwrap())?;
        let shape: Vec<usize> = header_info["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let data_offsets = header_info["data_offsets"].as_array().unwrap();

        let offset = data_offsets[0].as_u64().unwrap() as usize + self.header_size + 8; // Add 8 to account for the header size bytes
        let length = data_offsets[1].as_u64().unwrap() as usize - data_offsets[0].as_u64().unwrap() as usize;

        let shape_product: usize = shape.iter().product();
        if shape_product * dtype.size_in_bytes() != length {
            return Err(FastTensorsError::InvalidData(format!(
                "Tensor shape doesn't match storage size: {}",
                key
            )));
        }

        let mut tensor = Tensor::zeros(shape, dtype, device)?;

        self.file_handle.lock().unwrap().load(&mut tensor, device, offset, length)
            .await.map_err(|e| {
                FastTensorsError::SafeTensors(SafeTensorError::TensorNotFound(e.to_string()))
            })?;

        let _ = device.synchronize(); //TODO review this
        Ok(tensor)
    }
}

pub fn convert_dtype(dt: &str) -> Result<DType, FastTensorsError> {
    match dt {
        "I32"  => Ok(DType::I32),
        "I16"  => Ok(DType::I16,),
        "F16"  => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "F32"  => Ok(DType::F32),
        _      => Err(FastTensorsError::UnknownDtype(dt.to_string())),
    }
}

pub fn cleanup() {
    if let Some(cache) = STFILE_CACHE.get() {
        cache.clear();
    }
    if let Some(cache) = TENSOR_CACHE.get() {
        cache.clear();
    }
    if let Some(cache) = CONTEXT_CACHE.get() {
        cache.clear();
    }
    let _ = mem::free_pinned_memory().map_err(|e| {
        eprintln!("Error freeing pinned memory: {:?}", e);
    });
}

impl Drop for FastTensorFile {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            eprintln!("Error during FastTensorFile cleanup: {:?}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_fast_mode_first_key() -> Result<(), FastTensorsError> {
        // Initialize caches
        init_caches();

        // Path to your safetensors file
        let model_path = "/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00002-of-00005.safetensors";

        // Create a FastTensorFile instance in fast mode
        let fast_tensor_file = FastTensorFile::new(model_path, true, None).await?;

        // Get the first key
        //let first_key = fast_tensor_file.get_keys().first().unwrap().clone();
        let first_key = "model.layers.10.mlp.down_proj.q_invperm".to_string();

        // Get the tensor for the first key using fast mode
        let device = Device::cuda_if_available(0)?;
        let tensor = fast_tensor_file.get_tensor(&first_key, &device, false, None).await?;

        // Check if the first value is 4643
        let first_value = tensor.to_vec1::<i32>()?[0];
        //assert_eq!(first_value, 1618246796, "The first value should be 1618246796");

        // Cleanup
        cleanup();

        Ok(())
    }
}