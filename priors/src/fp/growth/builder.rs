use super::tree::FPTree;
use numpy::ndarray::{ArrayView2, Axis};
use std::collections::{HashMap, HashSet};

pub fn build_fp_tree(transactions: ArrayView2<i32>, min_support: f64) -> (FPTree, Vec<usize>) {
    let num_transactions = transactions.shape()[0];
    let min_count = (min_support * num_transactions as f64).ceil() as usize;

    let mut frequent_items: Vec<(usize, usize)> = transactions
        .sum_axis(Axis(0))
        .iter()
        .enumerate()
        .filter_map(|(idx, &count)| (count as usize >= min_count).then_some((idx, count as usize)))
        .collect();

    frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let ordered_items: Vec<usize> = frequent_items.iter().map(|&(item, _)| item).collect();
    let mut fp_tree = FPTree::new();

    for tx_idx in 0..num_transactions {
        let tx_items: Vec<usize> = ordered_items
            .iter()
            .filter(|&&item| transactions[[tx_idx, item]] != 0)
            .copied()
            .collect();

        if !tx_items.is_empty() {
            fp_tree.insert_transaction(&tx_items, &vec![1; tx_items.len()]);
        }
    }

    (fp_tree, ordered_items)
}

pub fn build_conditional_fp_tree(prefix_paths: &[(Vec<usize>, usize)], min_count: usize) -> FPTree {
    let mut item_counts: HashMap<usize, usize> = HashMap::new();

    for (path, count) in prefix_paths {
        for &item in path {
            *item_counts.entry(item).or_insert(0) += count;
        }
    }

    let frequent_items: HashSet<usize> = item_counts
        .iter()
        .filter_map(|(&item, &count)| (count >= min_count).then_some(item))
        .collect();

    let mut conditional_tree = FPTree::new();

    for (path, count) in prefix_paths {
        let filtered_path: Vec<usize> = path
            .iter()
            .copied()
            .filter(|item| frequent_items.contains(item))
            .collect();

        if !filtered_path.is_empty() {
            conditional_tree.insert_transaction(&filtered_path, &vec![*count; filtered_path.len()]);
        }
    }

    conditional_tree
}

pub fn get_conditional_frequent_items(tree: &FPTree, min_count: usize) -> Vec<usize> {
    let mut frequent_items: Vec<(usize, usize)> = tree
        .header_table
        .iter()
        .filter_map(|(&item, node_indices)| {
            let count: usize = node_indices.iter().map(|&idx| tree.nodes[idx].count).sum();
            (count >= min_count).then_some((item, count))
        })
        .collect();

    frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    frequent_items.into_iter().map(|(item, _)| item).collect()
}
