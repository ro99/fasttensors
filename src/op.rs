


/// Copy a tensor to the GPU
#[derive(Clone)]
pub struct CudaCopyToDevice<T>(Arc<CudaDevice>, PhantomData<T>);

