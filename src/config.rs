pub mod config {
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, Serialize)]
    pub struct ModelConfig {
        // Fields from config.json
    }

    impl ModelConfig {
        pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
            // Implementation here
        }

        // Other methods as needed
    }
}