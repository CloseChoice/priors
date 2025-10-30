use numpy::ndarray::ArrayView2;

pub struct StreamConfig {
    pub chunk_size: usize,
    pub min_support: f64,
    pub memory_limit: Option<usize>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            min_support: 0.01,
            memory_limit: None,
        }
    }
}

impl StreamConfig {
    pub fn new(chunk_size: usize, min_support: f64) -> Self {
        Self {
            chunk_size,
            min_support,
            memory_limit: None,
        }
    }

    pub fn with_memory_limit(mut self, limit_bytes: usize) -> Self {
        self.memory_limit = Some(limit_bytes);
        self
    }
}

pub trait ChunkProcessor {
    fn process_chunk(&mut self, chunk: ArrayView2<i32>) -> Result<(), ProcessingError>;
    fn finalize(&mut self) -> Result<(), ProcessingError>;
}

#[derive(Debug, Clone)]
pub enum ProcessingError {
    MemoryExceeded { current: usize, limit: usize },
    InvalidChunk(String),
    ProcessingFailed(String),
}

impl std::fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingError::MemoryExceeded { current, limit } => {
                write!(f, "Memory exceeded: {} / {} bytes", current, limit)
            }
            ProcessingError::InvalidChunk(msg) => write!(f, "Invalid chunk: {}", msg),
            ProcessingError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
        }
    }
}

impl std::error::Error for ProcessingError {}

pub fn split_into_chunks<T>(data: &[T], chunk_size: usize) -> Vec<&[T]> {
    data.chunks(chunk_size).collect()
}
