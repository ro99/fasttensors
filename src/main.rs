use std::path::PathBuf;
use candle_core::Device;
use fasttensors::{cleanup, init_caches, FastTensorFile};
use std::time::Instant;
use std::collections::HashMap;
use candle_core::DType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        init_caches();

        let stfiles = vec![
            "/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00001-of-00005.safetensors",
            "/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00002-of-00005.safetensors",
            "/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00003-of-00005.safetensors",
            "/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00004-of-00005.safetensors",
            "/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00005-of-00005.safetensors",
        ];

        let device = Device::cuda_if_available(0)?;

        for stfile in stfiles {
            println!("Starting benchmark for file: {}", stfile); 
            let test_file = PathBuf::from(stfile);

            // Open the FastTensorFile
            let fast_tensor_file = FastTensorFile::new(&test_file, true, None).await?;

            // Get all keys
            let keys: Vec<String> = fast_tensor_file.get_keys();

            // Benchmark each tensor
            let mut tensors1 = HashMap::new();
            let mut tensors2 = HashMap::new();

            let start_fast = Instant::now();
            for key in &keys {
                let tensor = fast_tensor_file.get_tensor(key, &device, false, None).await?;
                tensors1.insert(key.clone(), tensor);
            }
            let duration_fast = start_fast.elapsed();
            println!("Total fast method took: {:?}", duration_fast);

            let start_slow = Instant::now();
            for key in &keys {
                let tensor = fast_tensor_file.get_tensor(key, &device, true, None).await?;
                tensors2.insert(key.clone(), tensor);
            }
            let duration_slow = start_slow.elapsed();
            println!("Total slow method took: {:?}", duration_slow);

            for key in keys {
                let tensor_fast = tensors1.get(&key).unwrap();
                let tensor_slow = tensors2.get(&key).unwrap();

                // Check the dtype and shape of the tensors
                let dtype_fast = tensor_fast.dtype();
                let dtype_slow = tensor_slow.dtype();
                let shape_fast = tensor_fast.shape();
                let shape_slow = tensor_slow.shape();

                if dtype_fast != dtype_slow || shape_fast != shape_slow {
                    println!("Tensors have different dtypes or shapes for key: {}", key);
                    return Err(std::io::Error::new(std::io::ErrorKind::Other, "Tensors have different dtypes or shapes").into());
                }

                // Convert tensors to vectors based on dtype
                match dtype_fast {
                    DType::F16 => {
                        let vec_fast = tensor_fast.to_vec1::<half::f16>()?;
                        let vec_slow = tensor_slow.to_vec1::<half::f16>()?;

                        if vec_fast != vec_slow {
                            println!("Vectors are different for key: {}", key);
                            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Vectors are different").into());
                        } else {
                            println!("Vectors are the same for key: {}", key);
                        }
                    },
                    DType::I32 => {
                        let vec_fast = tensor_fast.to_vec1::<i32>()?;
                        let vec_slow = tensor_slow.to_vec1::<i32>()?;

                        if vec_fast != vec_slow {
                            println!("Vectors are different for key: {}", key);
                            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Vectors are different").into());
                        } else {
                            println!("Vectors are the same for key: {}", key);
                        }
                    },
                    // Add more cases as needed for other dtypes
                    _ => {
                        println!("Unsupported dtype for key: {}", key);
                        return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported dtype").into());
                    }
                }
            }
            println!("Finished benchmarking file: {}", stfile);
        }

        // Cleanup
        cleanup();

        Ok(())
    })
}
