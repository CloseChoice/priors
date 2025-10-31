use std::collections::HashMap;
use super::super::growth::tree::FPTree;

/// State for streaming FP-Growth processing
#[derive(Debug)]
pub struct StreamingState {
    /// Item frequency counts during counting phase
    pub item_counts: HashMap<usize, usize>,
    /// Total number of transactions processed
    pub num_transactions: usize,
    /// Frequent items after finalization (sorted by frequency descending)
    pub frequent_items: Vec<usize>,
    /// Minimum support threshold
    pub min_support: Option<f64>,
    /// The FP-Tree being built incrementally
    pub fp_tree: Option<FPTree>,
    /// Processing phase
    pub phase: ProcessingPhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingPhase {
    Counting,
    CountingFinalized,
    Building,
    ReadyToMine,
}

impl StreamingState {
    pub fn new() -> Self {
        Self {
            item_counts: HashMap::new(),
            num_transactions: 0,
            frequent_items: Vec::new(),
            min_support: None,
            fp_tree: None,
            phase: ProcessingPhase::Counting,
        }
    }

    /// Add item counts from a transaction batch
    pub fn add_counts(&mut self, transactions: &[Vec<usize>]) {
        for transaction in transactions {
            self.num_transactions += 1;
            for &item in transaction {
                *self.item_counts.entry(item).or_insert(0) += 1;
            }
        }
    }

    /// Finalize counting phase and determine frequent items
    pub fn finalize_counts(&mut self, min_support: f64) -> Result<(), String> {
        if self.phase != ProcessingPhase::Counting {
            return Err(format!("Cannot finalize counts in phase {:?}", self.phase));
        }

        self.min_support = Some(min_support);
        let min_count = (min_support * self.num_transactions as f64) as usize;

        // Filter frequent items and sort by frequency (descending)
        let mut frequent: Vec<(usize, usize)> = self.item_counts
            .iter()
            .filter(|&(_, &count)| count >= min_count)
            .map(|(&item, &count)| (item, count))
            .collect();

        frequent.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        self.frequent_items = frequent.into_iter().map(|(item, _)| item).collect();

        self.phase = ProcessingPhase::CountingFinalized;
        Ok(())
    }

    /// Initialize FP-Tree for building phase
    pub fn init_tree(&mut self) -> Result<(), String> {
        if self.phase != ProcessingPhase::CountingFinalized {
            return Err(format!("Cannot init tree in phase {:?}", self.phase));
        }

        self.fp_tree = Some(FPTree::new());
        self.phase = ProcessingPhase::Building;
        Ok(())
    }

    /// Get the rank of an item (for sorting transactions)
    pub fn get_item_rank(&self, item: usize) -> Option<usize> {
        self.frequent_items.iter().position(|&x| x == item)
    }

    /// Check if an item is frequent
    pub fn is_frequent(&self, item: usize) -> bool {
        self.frequent_items.contains(&item)
    }

    /// Complete building phase
    pub fn finalize_building(&mut self) -> Result<(), String> {
        if self.phase != ProcessingPhase::Building {
            return Err(format!("Cannot finalize building in phase {:?}", self.phase));
        }

        self.phase = ProcessingPhase::ReadyToMine;
        Ok(())
    }
}
