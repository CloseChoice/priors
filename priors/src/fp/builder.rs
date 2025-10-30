use super::tree::FPTree;
use numpy::ndarray::{ArrayView2, Axis};
use std::collections::HashMap;

pub fn build_fp_tree(transactions: ArrayView2<i32>, min_support: f64) -> (FPTree, Vec<usize>) {
    let num_transactions = transactions.shape()[0];
    let min_count = (min_support * num_transactions as f64).ceil() as usize;

    let item_counts = transactions.sum_axis(Axis(0));
    let mut frequent_items: Vec<(usize, usize)> = item_counts
        .iter()
        .enumerate()
        .filter_map(|(idx, &count)| {
            if count as usize >= min_count {
                Some((item, count as usize))
            } else {
                None
            }
        })
        .collect();

    frequent_items.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let ordered_items: Vec<usize> = frequent_items.iter().map(|&(item, _)| item).collect();
    let mut fp_tree = FPTree::new();

    for tx_idx in 0..num_transactions {
        let mut tx_items: Vec<usize> = Vec::new();

        for &item in &ordered_items {
            if transactions[[tx_idx, item]] != 0 {
                tx_items.push(item);
            }
        }

        if !tx_items.is_empty() {
            let counts = vec![1; tx_items.len()];
            fp_tree.insert_transaction(&tx_items, &counts);
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

    let frequent_items: Vec<usize> = item_counts
        .iter()
        .filter_map(
            |(&item, &count)| {
                if count >= min_count { Some(item) } else { None }
            },
        )
        .collect();

    let mut conditional_tree = FPTree::new();

    for (path, count) in prefix_paths {
        let filtered_path: Vec<usize> = path
            .iter()
            .filter(|&&item| frequent_items.contains(&item))
            .cloned()
            .collect();

        if !filtered_path.is_empty() {
            let counts = vec![*count; filtered_path.len()];
            conditional_tree.insert_transaction(&filtered_path, &counts);
        }
    }

    conditional_tree
}

pub fn get_conditional_frequent_items(tree: &FPTree, min_count: usize) -> Vec<usize> {
    let mut item_counts: HashMap<usize, usize> = HashMap::new();

    for (&item, node_indices) in &tree.header_table {
        let total_count: usize = node_indices.iter().map(|&idx| tree.nodes[idx].count).sum();
        item_counts.insert(item, total_count);
    }

    let mut frequent_items: Vec<(usize, usize)> = item_counts
        .iter()
        .filter_map(|(&item, &count)| {
            if count >= min_count {
                Some((item, count))
            } else {
                None
            }
        })
        .collect();

    frequent_items.sort_by(|a, b| b.1.cmp(&a.1));

    frequent_items.into_iter().map(|(item, _)| item).collect()
}
