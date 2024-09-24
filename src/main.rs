use std::path::PathBuf;
use candle_core::Device;
use fasttensors::{cleanup, init_caches, FastTensorFile};
use std::time::Instant;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        let fast_tensor_file = FastTensorFile::new(&test_file, true, None)?;

        // Get all keys
        let keys: Vec<String> = fast_tensor_file.get_keys();

        // Benchmark each tensor
        let mut tensors1 = HashMap::new();
        let mut tensors2 = HashMap::new();

        let start_fast = Instant::now();
        for key in &keys {
            let tensor = fast_tensor_file.get_tensor(key, &device, false, None)?;
            tensors1.insert(key.clone(), tensor);
        }
        let duration_fast = start_fast.elapsed();
        println!("Total fast method took: {:?}", duration_fast);


        let start_slow = Instant::now();
        for key in &keys {
            let tensor = fast_tensor_file.get_tensor(key, &device, true, None)?;
            tensors2.insert(key.clone(), tensor);
        }
        let duration_slow = start_slow.elapsed();
        println!("Total slow method took: {:?}", duration_slow);

        for key in keys {
            let tensor_fast = tensors1.get(&key).unwrap();
            let tensor_slow = tensors2.get(&key).unwrap();
            if tensor_fast.to_vec1::<i32>()? != tensor_slow.to_vec1::<i32>()? {
                println!("Vectors are different for key: {}", key);
            } else {
                println!("Vectors are the same for key: {}", key);
            }
        }

        println!("Finished benchmarking file: {}", stfile);
    }

    // Cleanup
    cleanup();

    Ok(())
}
