use std::path::Path;
use candle_core::{DType, Device, DeviceLocation, Error, Tensor};
use cache::GLOBAL_CACHE;
use std::collections::HashMap;
//mod config;
mod cache;


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
    pub fn open<P: AsRef<Path>>(filename: P, fast: bool) -> Result<Self, Error> {
        // Implementation
        todo!()
    }

    pub fn close(&mut self) -> Result<(), Error> {
        // Implementation
        todo!()
    }

    pub fn get_dict(&self) -> &serde_json::Value {   // or HashMap<String, TensorMetadata>
        // Implementation
        todo!()
    }

    pub fn get_metadata(&self) -> Option<&serde_json::Value> {
        // Implementation
        todo!()
    }

    pub fn measure(&self, key: &str) -> usize {                   // Result<usize, Error>  ?
        // Implementation
        todo!()
    }

    pub fn get_tensor(&self, key: &String, device: &Device, not_fast: bool, cached: bool, out_dtype: Option<DType>) -> Result<Tensor, Error> {
        let _ = device.synchronize();



        let key = if let Some(remap) = &self.tensor_remap {
            remap.get(key).unwrap_or(key)
        } else {
            key
        };
        let cache_key = format!("{}::{}::{}", self.filename, key, device_id(device));
        if cached {
            if let Some(cached_tensor) = GLOBAL_CACHE.get(&cache_key) {
                return Ok(cached_tensor.as_ref().clone());
            }
        }




        // Add to cache if caching is enabled
        //if cached {
        //    GLOBAL_CACHE.insert(cache_key, tensor.clone());
        //}
        
        todo!()
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


/* 
pub enum Device {
    CPU,
    CUDA(usize),
}

pub struct TensorMetadata {
    // Fields like shape, dtype, etc.
}

pub enum Error {
    // Various error types
}*/


fn device_id(device: &Device) -> String {
    match device.location() {
        DeviceLocation::Cpu => "cpu".to_string(),
        DeviceLocation::Cuda { gpu_id } => format!("cuda:{}", gpu_id),
        DeviceLocation::Metal { gpu_id } => format!("metal:{}", gpu_id),
    }
}