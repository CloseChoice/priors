use super::builder::{build_conditional_fp_tree, build_fp_tree, get_conditional_frequent_items};
use crate::fp::utils::FrequentLevel;
use super::tree::FPTree;
use numpy::ndarray::ArrayView2;
use rayon::prelude::*;

pub fn fp_growth_algorithm(transactions: ArrayView2<i32>, min_support: f64) -> Vec<FrequentLevel> {
    let num_transactions = transactions.shape()[0];
    let (fp_tree, frequent_items) = build_fp_tree(transactions, min_support);
    fp_growth_recursive(&fp_tree, &frequent_items, &[], min_support, num_transactions)
}

fn fp_growth_recursive(
    fp_tree: &FPTree,
    frequent_items: &[usize],
    alpha: &[usize],
    min_support: f64,
    num_transactions: usize,
) -> Vec<FrequentLevel> {
    let min_count = (min_support * num_transactions as f64) as usize;

    if fp_tree.has_single_path() {
        let mut result = Vec::new();
        let path = fp_tree.get_single_path();
        for i in 1..=path.len() {
            generate_combinations(&path, i, alpha, &mut result);
        }
        return result;
    }

    let mut results: Vec<Vec<FrequentLevel>> = frequent_items
        .par_iter()
        .rev()
        .filter_map(|&item| {
            let support = fp_tree.header_table.get(&item)?
                .iter()
                .map(|&idx| fp_tree.nodes[idx].count)
                .sum::<usize>();

            if support < min_count {
                return None;
            }

            let mut new_pattern = alpha.to_vec();
            new_pattern.push(item);
            let mut result = vec![FrequentLevel::new(new_pattern.len())];
            result[0].add_itemset(new_pattern.clone());

            let prefix_paths = fp_tree.get_prefix_paths(item);
            if !prefix_paths.is_empty() {
                let cond_tree = build_conditional_fp_tree(&prefix_paths, min_count);
                let cond_items = get_conditional_frequent_items(&cond_tree, min_count);

                if !cond_items.is_empty() {
                    result.extend(fp_growth_recursive(&cond_tree, &cond_items, &new_pattern, min_support, num_transactions));
                }
            }

            Some(result)
        })
        .collect();

    // Merge results sequentially
    let mut merged = Vec::new();
    for item_results in results.iter_mut() {
        for level in item_results.drain(..) {
            let size = level.itemset_size;
            while merged.len() < size {
                merged.push(FrequentLevel::new(merged.len() + 1));
            }
            merged[size - 1].storage.items.extend_from_slice(&level.storage.items);
            merged[size - 1].storage.offsets.extend_from_slice(&level.storage.offsets);
        }
    }
    merged
}

fn generate_combinations(
    path: &[(usize, usize)],
    k: usize,
    alpha: &[usize],
    result: &mut Vec<FrequentLevel>,
) {
    if k == 0 || k > path.len() {
        return;
    }

    let mut current = Vec::with_capacity(k);
    generate_comb_recursive(path, k, 0, &mut current, alpha, result);
}

fn generate_comb_recursive(
    path: &[(usize, usize)],
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    alpha: &[usize],
    result: &mut Vec<FrequentLevel>,
) {
    if current.len() == k {
        let mut pattern = Vec::with_capacity(alpha.len() + k);
        pattern.extend_from_slice(alpha);
        pattern.extend(current.iter().map(|&idx| path[idx].0));

        while result.len() < pattern.len() {
            result.push(FrequentLevel::new(result.len() + 1));
        }
        result[pattern.len() - 1].add_itemset(pattern);
        return;
    }

    for i in start..path.len() {
        current.push(i);
        generate_comb_recursive(path, k, i + 1, current, alpha, result);
        current.pop();
    }
}
