pub struct StreamConfig {
    pub chunk_size: usize,
    pub min_support: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self { chunk_size: 1000, min_support: 0.01 }
    }
}

#[derive(Debug, Clone)]
pub enum ProcessingError {
    ProcessingFailed(String),
}

impl std::fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
        }
    }
}

impl std::error::Error for ProcessingError {}
