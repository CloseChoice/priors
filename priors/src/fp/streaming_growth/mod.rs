pub mod processor;
pub mod state;

pub use processor::{build_pass, count_pass, finalize_building, finalize_counts, mine_patterns};
pub use state::{ProcessingPhase, StreamingState};
