use std::path::PathBuf;
use candle_core::Device;
use fasttensors::{cleanup, init_caches, FastTensorFile};
use std::time::Instant;
use std::collections::HashMap;
use candle_core::DType;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;


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

            let mut total_keys = 0;
            let mut matching_keys = 0;

            for key in keys {
                total_keys += 1;
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
                        let vec_fast = match tensor_fast.shape().dims().len() {
                            1 => vec![tensor_fast.to_vec1::<half::f16>()?],
                            2 => tensor_fast.to_vec2::<half::f16>()?,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported tensor shape").into()),
                        };
                        let vec_slow = match tensor_slow.shape().dims().len() {
                            1 => vec![tensor_slow.to_vec1::<half::f16>()?],
                            2 => tensor_slow.to_vec2::<half::f16>()?,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported tensor shape").into()),
                        };

                        let epsilon = 1e-3; // Adjust this value as needed
                        let are_equal = vec_fast.iter().zip(vec_slow.iter()).all(|(a, b)| {
                            (a.iter().zip(b.iter()).all(|(x, y)| (x.to_f32() - y.to_f32()).abs() < epsilon))
                        });

                        if !are_equal {
                            println!("Vectors are different for key: {}", key);
                            // Find and print the first difference
                            for (i, (fast_val, slow_val)) in vec_fast.iter().zip(vec_slow.iter()).enumerate() {
                                if fast_val != slow_val {
                                    println!("First difference at index {}: fast = {:?}, slow = {:?}", i, fast_val, slow_val);
                                    // Print surrounding values for context
                                    break;
                                }
                            }
                        } else {
                            matching_keys += 1;
                        }
                    },
                    DType::I32 => {
                        let vec_fast = match tensor_fast.shape().dims().len() {
                            1 => vec![tensor_fast.to_vec1::<i32>()?],
                            2 => tensor_fast.to_vec2::<i32>()?,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported tensor shape").into()),
                        };
                        let vec_slow = match tensor_slow.shape().dims().len() {
                            1 => vec![tensor_slow.to_vec1::<i32>()?],
                            2 => tensor_slow.to_vec2::<i32>()?,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported tensor shape").into()),
                        };

                        println!("Fast vector length: {}", vec_fast[0].len());
                        println!("Slow vector length: {}", vec_slow[0].len());

                        // Compare raw pointers
                        println!("Fast vector pointer: {:p}", vec_fast[0].as_ptr());
                        println!("Slow vector pointer: {:p}", vec_slow[0].as_ptr());

                        // Hash comparison
                        let mut fast_hasher = DefaultHasher::new();
                        vec_fast[0].hash(&mut fast_hasher);
                        let fast_hash = fast_hasher.finish();

                        let mut slow_hasher = DefaultHasher::new();
                        vec_slow[0].hash(&mut slow_hasher);
                        let slow_hash = slow_hasher.finish();

                        println!("Fast vector hash: {:x}", fast_hash);
                        println!("Slow vector hash: {:x}", slow_hash);

                        if vec_fast != vec_slow {
                            println!("Vectors are different for key: {}", key);
                            
                            // Full vector comparison
                            let mut differences = 0;
                            for (i, (fast_val, slow_val)) in vec_fast[0].iter().zip(vec_slow[0].iter()).enumerate() {
                                if fast_val != slow_val {
                                    differences += 1;
                                    if differences <= 5 {  // Print only the first 5 differences
                                        println!("Difference at index {}: fast = {}, slow = {}", i, fast_val, slow_val);
                                    }
                                }
                            }
                            println!("Total differences: {}", differences);

                            if differences == 0 {
                                println!("No differences found in full element-wise comparison, but vectors are not equal.");
                                println!("First 10 elements of fast vector: {:?}", &vec_fast[0][..10]);
                                println!("First 10 elements of slow vector: {:?}", &vec_slow[0][..10]);
                            }
                        } else {
                            matching_keys += 1;
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
            println!("Matching keys: {}/{}", matching_keys, total_keys);
            println!("-----------------------------------");
        }

        // Cleanup
        cleanup();

        Ok(())
    })
}
