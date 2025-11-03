pub mod state;
pub mod processor;

pub use state::{StreamingState, ProcessingPhase};
pub use processor::{count_pass, finalize_counts, build_pass, finalize_building, mine_patterns};
