use std::path::PathBuf;
use candle_core::{Device, Tensor};
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
            //"/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00003-of-00005.safetensors",
            //"/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00004-of-00005.safetensors",
            //"/home/rodrigo/AI/Models/Exllama/Cohere-Aya23-35B/Cohere-aya-23-35B-8.0bpw-h8-exl2/output-00005-of-00005.safetensors",
        ];

        let device = Device::cuda_if_available(0)?;

        // Process all files in fast mode
        println!("Processing files in fast mode:");
        let mut fast_times = Vec::new();
        for stfile in &stfiles {
            let (duration, _) = process_file(stfile, &device, false).await?;
            fast_times.push(duration);
            println!("File: {}, Fast mode took: {:?}", stfile, duration);
        }

        // Cleanup after fast mode
        cleanup();

         // Process all files in slow mode
        println!("\nProcessing files in slow mode:");
        let mut slow_times = Vec::new();
        for stfile in &stfiles {
            let (duration, _) = process_file(stfile, &device, true).await?;
            slow_times.push(duration);
            println!("File: {}, Slow mode took: {:?}", stfile, duration);
        }

        // Compare results
        for (i, (fast_time, slow_time)) in fast_times.iter().zip(slow_times.iter()).enumerate() {
            println!("\nFile {}: ", stfiles[i]);
            println!("  Fast mode: {:?}", fast_time);
            println!("  Slow mode: {:?}", slow_time);
            println!("  Speed-up: {:.2}x", slow_time.as_secs_f64() / fast_time.as_secs_f64());
        }

        // Optional: Compare tensors if needed
        // for i in 0..stfiles.len() {
        //     compare_tensors(&fast_tensors[i], &slow_tensors[i], &stfiles[i])?;
        // }

        Ok(())
    })
}

async fn process_file(stfile: &str, device: &Device, slow_mode: bool) -> Result<(std::time::Duration, HashMap<String, Tensor>), Box<dyn std::error::Error>> {
    let test_file = PathBuf::from(stfile);
    let fast_tensor_file = FastTensorFile::new(&test_file, true, None).await?;
    let keys: Vec<String> = fast_tensor_file.get_keys();

    let mut tensors = HashMap::new();
    let start = Instant::now();

    for key in &keys {
        let tensor = fast_tensor_file.get_tensor(key, device, slow_mode, None).await?;
        tensors.insert(key.clone(), (*tensor).clone());
    }

    let duration = start.elapsed();
    Ok((duration, tensors))
}

fn compare_tensors(
    keys: &[String],
    tensors1: &HashMap<String, Tensor>,
    tensors2: &HashMap<String, Tensor>,
    stfile: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut total_keys = 0;
    let mut matching_keys = 0;

    for key in keys {
        total_keys += 1;
        let tensor_fast = tensors1.get(key).unwrap();
        let tensor_slow = tensors2.get(key).unwrap();

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

                if vec_fast != vec_slow {
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
            DType::I16 => {
                let vec_fast = match tensor_fast.shape().dims().len() {
                    1 => vec![tensor_fast.to_vec1::<i16>()?],
                    2 => tensor_fast.to_vec2::<i16>()?,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported tensor shape").into()),
                };
                let vec_slow = match tensor_slow.shape().dims().len() {
                    1 => vec![tensor_slow.to_vec1::<i16>()?],
                    2 => tensor_slow.to_vec2::<i16>()?,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported tensor shape").into()),
                };

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
                println!("Unsupported dtype {:?} for key: {}", dtype_fast, key);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported dtype").into());
            }
        }
    }
    println!("Finished benchmarking file: {}", stfile);
    println!("Matching keys: {}/{}", matching_keys, total_keys);
    println!("-----------------------------------");
    Ok(())
}