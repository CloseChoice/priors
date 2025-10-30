use super::builder::{build_conditional_fp_tree, build_fp_tree, get_conditional_frequent_items};
use super::combinations::generate_combinations_from_path;
use super::storage::FrequentLevel;
use super::tree::FPTree;
use numpy::ndarray::ArrayView2;
use rayon::prelude::*;

pub fn fp_growth_algorithm(transactions: ArrayView2<i32>, min_support: f64) -> Vec<FrequentLevel> {
    let num_transactions = transactions.shape()[0];

    let (fp_tree, frequent_items) = build_fp_tree(transactions, min_support);
    let alpha = Vec::new();
    fp_growth_recursive_parallel(
        &fp_tree,
        &frequent_items,
        &alpha,
        min_support,
        num_transactions,
    )
}

pub fn fp_growth_recursive_parallel(
    fp_tree: &FPTree,
    frequent_items: &[usize],
    alpha: &[usize],
    min_support: f64,
    num_transactions: usize,
) -> Vec<FrequentLevel> {
    let min_count = (min_support * num_transactions as f64) as usize;
    let mut local_result = Vec::new();

    if fp_tree.has_single_path() {
        let path = fp_tree.get_single_path();
        let alpha_vec = alpha.to_vec();

        for i in 1..=path.len() {
            generate_combinations_from_path(&path, i, &alpha_vec, &mut local_result);
        }
        return local_result;
    }

    let parallel_results: Vec<Vec<FrequentLevel>> = frequent_items
        .par_iter()
        .rev()
        .map(|&item| {
            let mut item_result = Vec::new();

            let mut new_pattern = alpha.to_vec();
            new_pattern.push(item);

            let item_support = if let Some(node_indices) = fp_tree.header_table.get(&item) {
                node_indices
                    .iter()
                    .map(|&idx| fp_tree.nodes[idx].count)
                    .sum::<usize>()
            } else {
                0
            };

            if item_support >= min_count {
                add_pattern_to_result(&new_pattern, &mut item_result);

                let prefix_paths = fp_tree.get_prefix_paths(item);

                if !prefix_paths.is_empty() {
                    let conditional_tree = build_conditional_fp_tree(&prefix_paths, min_count);

                    let conditional_frequent_items =
                        get_conditional_frequent_items(&conditional_tree, min_count);

                    if !conditional_frequent_items.is_empty() {
                        let recursive_results = fp_growth_recursive_parallel(
                            &conditional_tree,
                            &conditional_frequent_items,
                            &new_pattern,
                            min_support,
                            num_transactions,
                        );

                        item_result.extend(recursive_results);
                    }
                }
            }

            item_result
        })
        .collect();

    for item_results in parallel_results {
        local_result.extend(item_results);
    }

    local_result
}

pub fn add_pattern_to_result(pattern: &[usize], result: &mut Vec<FrequentLevel>) {
    let pattern_size = pattern.len();

    while result.len() < pattern_size {
        result.push(FrequentLevel::new(result.len() + 1));
    }

    if pattern_size > 0 {
        result[pattern_size - 1].add_itemset(pattern.to_vec());
    }
}
