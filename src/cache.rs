use candle_core::{Device, Tensor};
use lru::LruCache;
use once_cell::sync::Lazy;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

const CACHE_CAPACITY: usize = 4;
pub static GLOBAL_CACHE: Lazy<TensorCache> = Lazy::new(|| TensorCache::new(CACHE_CAPACITY));

pub struct TensorCache {
    cache: Mutex<LruCache<String, Arc<Tensor>>>,
}

impl TensorCache {
    pub fn new(capacity: usize) -> Self {
        TensorCache {
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(capacity).unwrap())),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<Tensor>> {
        let mut cache = self.cache.lock().unwrap();
        cache.get(key).cloned()
    }

    pub fn insert(&self, key: String, tensor: Tensor) {
        let tensor = Arc::new(tensor);
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, tensor);
    }
}
