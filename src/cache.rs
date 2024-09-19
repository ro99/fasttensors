use candle_core::Tensor;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use crate::STFile;

pub struct Cache<T> {
    cache: Mutex<LruCache<String, Arc<T>>>,
}

impl<T> Cache<T> {
    pub fn new(capacity: usize) -> Self {
        Cache {
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(capacity).unwrap())),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<T>> {
        let mut cache = self.cache.lock().unwrap();
        cache.get(key).cloned()
    }

    pub fn insert(&self, key: String, value: T) {
        let value = Arc::new(value);
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, value);
    }
}

pub type TensorCache = Cache<Tensor>;
pub type ContextCache = Cache<HashMap<String, Tensor>>;
pub type STFileCache = Cache<STFile>;

