use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::hash::Hash;

pub struct Cache<K, T> {
    cache: Mutex<LruCache<K, Arc<T>>>,
}

impl<K, T> Cache<K, T>
where
    K: Hash + Eq + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Cache {
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(capacity).unwrap())),
        }
    }

    pub fn get(&self, key: &K) -> Option<Arc<T>> {
        let mut cache = self.cache.lock().unwrap();
        cache.get(key).cloned()
    }

    pub fn insert<V>(&self, key: K, value: V) 
    where
        V: Into<Arc<T>>,
    {
        let value = value.into();
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, value);
    }

    pub fn pop_lru(&self) -> Option<(K, Arc<T>)> {
        let mut cache = self.cache.lock().unwrap();
        cache.pop_lru().map(|(k, v)| (k, v.clone()))
    }
}




