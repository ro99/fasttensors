use candle_core::cuda::cudarc::{
    driver::{sys::CUresult, CudaDevice, CudaFunction},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::{collections::hash_map::DefaultHasher, hash::Hasher, sync::Arc};

use crate::mem::get_cuda_lib;



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
