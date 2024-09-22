use candle_core::{DType, Error, Tensor, WithDType};

fn data_to_bytes<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let length = vs.len() * size_in_bytes;
    let capacity = vs.capacity() * size_in_bytes;
    let ptr = vs.as_mut_ptr() as *mut u8;
    // Don't run the destructor for Vec<T>
    std::mem::forget(vs);
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

pub(crate) fn serialize_tensor(buffer: &mut Vec<u8>, tensor: &Tensor) -> Result<(), Error> {
    let b_shape = tensor.dims();
    let tensor = tensor.flatten_all()?;

    let bias = match tensor.dtype() {
        DType::U8 => data_to_bytes::<u8>(tensor.to_vec1()?),
        DType::U32 => data_to_bytes::<u32>(tensor.to_vec1()?),
        DType::I64 => data_to_bytes::<i64>(tensor.to_vec1()?),
        DType::I16 => data_to_bytes::<i16>(tensor.to_vec1()?),
        DType::I32 => data_to_bytes::<i32>(tensor.to_vec1()?),
        DType::F16 => data_to_bytes::<half::f16>(tensor.to_vec1()?),
        DType::BF16 => data_to_bytes::<half::bf16>(tensor.to_vec1()?),
        DType::F32 => data_to_bytes::<f32>(tensor.to_vec1()?),
        DType::F64 => data_to_bytes::<f64>(tensor.to_vec1()?),
    };
    buffer.extend(&(bias.len() as u32).to_le_bytes());

    let dtype: u32 = match tensor.dtype() {
        DType::U8 => 0,
        DType::U32 => 1,
        DType::I16 => 2,
        DType::I32 => 3,
        DType::I64 => 4,
        DType::F16 => 5,
        DType::BF16 => 6,
        DType::F32 => 7,
        DType::F64 => 8,
    };
    buffer.extend(&dtype.to_le_bytes());

    // Shape
    buffer.extend((b_shape.len() as u32).to_le_bytes());
    for dim in b_shape {
        buffer.extend((*dim as u32).to_le_bytes());
    }

    buffer.extend(bias);

    Ok(())
}