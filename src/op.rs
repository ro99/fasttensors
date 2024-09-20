use candle_core::cuda::cudarc::{
    driver::{sys::CUresult, CudaDevice, CudaFunction},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::{collections::hash_map::DefaultHasher, hash::Hasher, sync::Arc};

use crate::mem::get_cuda_lib;

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

pub fn free_pinned_memory(ptr: *mut u8) -> Result<(), CUresult> {
    let cuda_lib = get_cuda_lib();
    unsafe {
        let result = cuda_lib.cuMemFreeHost(ptr as *mut std::ffi::c_void);
        if result != CUresult::CUDA_SUCCESS {
            return Err(result);
        }
    }
    Ok(())
}

pub fn compile_and_load_kernel(mut code: String, device: &Arc<CudaDevice>) -> CudaFunction {
    let name = format!("kernel_{}", hash(&code));
    code = code.replace("kernel", &name);
    if !device.has_func(&name, &name) {
        device
            .load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[name.clone().leak()],
            )
            .unwrap();
    }
    device.get_func(&name, &name).unwrap()
}

fn hash<T: std::hash::Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}
