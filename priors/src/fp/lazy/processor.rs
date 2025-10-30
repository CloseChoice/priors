use super::streaming::{ProcessingError, StreamConfig};
use crate::fp::mining::fp_growth_recursive_parallel;
use crate::fp::storage::FrequentLevel;
use crate::fp::tree::FPTree;
use numpy::ndarray::ArrayView2;
use std::collections::HashMap;

enum ProcessingPhase {
    Counting,
    Building,
}

pub struct LazyFPGrowth {
    item_counts: HashMap<usize, usize>,
    frequent_items: Vec<usize>,
    fp_tree: Option<FPTree>,
    total_transactions: usize,
    phase: ProcessingPhase,
    _config: StreamConfig,
}

impl LazyFPGrowth {
    pub fn new() -> Self {
        Self::with_config(StreamConfig::default())
    }

    pub fn with_config(config: StreamConfig) -> Self {
        Self {
            item_counts: HashMap::new(),
            frequent_items: Vec::new(),
            fp_tree: None,
            total_transactions: 0,
            phase: ProcessingPhase::Counting,
            _config: config,
        }
    }

    pub fn count_pass(&mut self, chunk: ArrayView2<i32>) {
        let num_transactions = chunk.shape()[0];
        let num_items = chunk.shape()[1];

        self.total_transactions += num_transactions;

        for tx_idx in 0..num_transactions {
            for item_idx in 0..num_items {
                if chunk[[tx_idx, item_idx]] != 0 {
                    *self.item_counts.entry(item_idx).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn finalize_counts(&mut self, min_support: f64) -> Vec<usize> {
        let min_count = (min_support * self.total_transactions as f64).ceil() as usize;

        let mut frequent_items: Vec<(usize, usize)> = self
            .item_counts
            .iter()
            .filter_map(|(&item, &count)| {
                if count >= min_count {
                    Some((item, count))
                } else {
                    None
                }
            })
            .collect();

        frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        self.frequent_items = frequent_items.iter().map(|&(item, _)| item).collect();
        self.phase = ProcessingPhase::Building;

        self.frequent_items.clone()
    }

    pub fn build_pass(&mut self, chunk: ArrayView2<i32>) {
        if self.fp_tree.is_none() {
            self.fp_tree = Some(FPTree::new());
        }

        let fp_tree = self.fp_tree.as_mut().unwrap();
        let num_transactions = chunk.shape()[0];

        for tx_idx in 0..num_transactions {
            let mut tx_items: Vec<usize> = Vec::new();

            for &item in &self.frequent_items {
                if chunk[[tx_idx, item]] != 0 {
                    tx_items.push(item);
                }
            }

            if !tx_items.is_empty() {
                let counts = vec![1; tx_items.len()];
                fp_tree.insert_transaction(&tx_items, &counts);
            }
        }
    }

    pub fn mine_patterns(&self, min_support: f64) -> Result<Vec<FrequentLevel>, ProcessingError> {
        let fp_tree = self.fp_tree.as_ref().ok_or_else(|| {
            ProcessingError::ProcessingFailed("FP-tree not built yet".to_string())
        })?;

        let alpha = Vec::new();
        let results = fp_growth_recursive_parallel(
            fp_tree,
            &self.frequent_items,
            &alpha,
            min_support,
            self.total_transactions,
        );

        Ok(results)
    }

    pub fn get_stats(&self) -> LazyStats {
        LazyStats {
            total_transactions: self.total_transactions,
            unique_items: self.item_counts.len(),
            frequent_items: self.frequent_items.len(),
            tree_nodes: self.fp_tree.as_ref().map(|t| t.nodes.len()).unwrap_or(0),
        }
    }

    pub fn reset(&mut self) {
        self.item_counts.clear();
        self.frequent_items.clear();
        self.fp_tree = None;
        self.total_transactions = 0;
        self.phase = ProcessingPhase::Counting;
    }
}

impl Default for LazyFPGrowth {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LazyStats {
    pub total_transactions: usize,
    pub unique_items: usize,
    pub frequent_items: usize,
    pub tree_nodes: usize,
}

pub fn lazy_fp_growth_from_chunks(
    chunks: impl Iterator<Item = ArrayView2<'static, i32>>,
    min_support: f64,
) -> Result<Vec<FrequentLevel>, ProcessingError> {
    let mut processor = LazyFPGrowth::new();

    let chunks_vec: Vec<_> = chunks.collect();

    for chunk in &chunks_vec {
        processor.count_pass(*chunk);
    }

    processor.finalize_counts(min_support);

    for chunk in &chunks_vec {
        processor.build_pass(*chunk);
    }

    processor.mine_patterns(min_support)
}
