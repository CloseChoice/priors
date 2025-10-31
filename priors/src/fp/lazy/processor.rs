use super::streaming::{ProcessingError, StreamConfig};
use crate::fp::mining::fp_growth_algorithm;
use crate::fp::storage::FrequentLevel;
use crate::fp::tree::FPTree;
use numpy::ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

pub struct LazyFPGrowth {
    item_counts: HashMap<usize, usize>,
    frequent_items: Vec<usize>,
    fp_tree: Option<FPTree>,
    total_transactions: usize,
    stored_transactions: Option<Array2<i32>>,
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
            stored_transactions: None,
            _config: config,
        }
    }

    pub fn store_transactions(&mut self, transactions: Array2<i32>) {
        self.stored_transactions = Some(transactions);
    }

    pub fn count_pass(&mut self, chunk: ArrayView2<i32>) {
        self.total_transactions += chunk.shape()[0];

        for tx_idx in 0..chunk.shape()[0] {
            for item_idx in 0..chunk.shape()[1] {
                if chunk[[tx_idx, item_idx]] != 0 {
                    *self.item_counts.entry(item_idx).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn finalize_counts(&mut self, min_support: f64) -> Vec<usize> {
        let min_count = (min_support * self.total_transactions as f64).ceil() as usize;

        let mut frequent_items: Vec<(usize, usize)> = self.item_counts.iter()
            .filter_map(|(&item, &count)| (count >= min_count).then_some((item, count)))
            .collect();

        frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        self.frequent_items = frequent_items.iter().map(|&(item, _)| item).collect();
        self.frequent_items.clone()
    }

    pub fn build_pass(&mut self, chunk: ArrayView2<i32>) {
        if self.fp_tree.is_none() {
            self.fp_tree = Some(FPTree::new());
        }

        let fp_tree = self.fp_tree.as_mut().unwrap();

        for tx_idx in 0..chunk.shape()[0] {
            let tx_items: Vec<usize> = self.frequent_items.iter()
                .filter(|&&item| chunk[[tx_idx, item]] != 0)
                .copied()
                .collect();

            if !tx_items.is_empty() {
                fp_tree.insert_transaction(&tx_items, &vec![1; tx_items.len()]);
            }
        }
    }

    pub fn mine_patterns(&self, min_support: f64) -> Result<Vec<FrequentLevel>, ProcessingError> {
        self.fp_tree.as_ref().ok_or_else(|| {
            ProcessingError::ProcessingFailed("FP-tree not built yet".to_string())
        })?;

        let transactions = self.stored_transactions.as_ref().ok_or_else(|| {
            ProcessingError::ProcessingFailed("No transactions stored".to_string())
        })?;

        Ok(fp_growth_algorithm(transactions.view(), min_support))
    }

    pub fn get_stats(&self) -> LazyStats {
        LazyStats {
            total_transactions: self.total_transactions,
            unique_items: self.item_counts.len(),
            frequent_items: self.frequent_items.len(),
            tree_nodes: self.fp_tree.as_ref().map(|t| t.nodes.len()).unwrap_or(0),
        }
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
