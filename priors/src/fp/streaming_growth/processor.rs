use super::state::{StreamingState, ProcessingPhase};
use super::super::growth::builder::{build_conditional_fp_tree, get_conditional_frequent_items};
use super::super::growth::tree::FPTree;
use super::super::utils::FrequentLevel;
use numpy::ndarray::ArrayView2;
use rayon::prelude::*;

/// Convert binary transaction matrix to list of item sets
fn matrix_to_transactions(transactions: ArrayView2<i32>) -> Vec<Vec<usize>> {
    let num_transactions = transactions.shape()[0];
    let num_items = transactions.shape()[1];

    (0..num_transactions)
        .map(|i| {
            (0..num_items)
                .filter(|&j| transactions[[i, j]] != 0)
                .collect()
        })
        .collect()
}

/// Process counting pass for streaming FP-Growth
pub fn count_pass(state: &mut StreamingState, transactions: ArrayView2<i32>) -> Result<(), String> {
    if state.phase != ProcessingPhase::Counting {
        return Err(format!("Cannot count in phase {:?}", state.phase));
    }

    let transaction_list = matrix_to_transactions(transactions);
    state.add_counts(&transaction_list);
    Ok(())
}

/// Finalize counting and determine frequent items
pub fn finalize_counts(state: &mut StreamingState, min_support: f64) -> Result<(), String> {
    state.finalize_counts(min_support)?;
    state.init_tree()?;
    Ok(())
}

/// Process building pass for streaming FP-Growth
pub fn build_pass(state: &mut StreamingState, transactions: ArrayView2<i32>) -> Result<(), String> {
    if state.phase != ProcessingPhase::Building {
        return Err(format!("Cannot build in phase {:?}", state.phase));
    }

    let transaction_list = matrix_to_transactions(transactions);

    // Build a map of item ranks to avoid borrowing conflicts
    let item_ranks: std::collections::HashMap<usize, usize> = state.frequent_items
        .iter()
        .enumerate()
        .map(|(rank, &item)| (item, rank))
        .collect();

    let fp_tree = state.fp_tree.as_mut()
        .ok_or("FP-Tree not initialized")?;

    for transaction in transaction_list {
        // Filter and sort transaction by frequency order
        let mut filtered: Vec<(usize, usize)> = transaction
            .iter()
            .filter_map(|&item| {
                item_ranks.get(&item).map(|&rank| (item, rank))
            })
            .collect();

        if filtered.is_empty() {
            continue;
        }

        filtered.sort_by_key(|&(_, rank)| rank);
        let sorted_items: Vec<usize> = filtered.iter().map(|&(item, _)| item).collect();
        let counts = vec![1; sorted_items.len()];

        fp_tree.insert_transaction(&sorted_items, &counts);
    }

    Ok(())
}

/// Finalize building phase
pub fn finalize_building(state: &mut StreamingState) -> Result<(), String> {
    state.finalize_building()
}

/// Mine patterns from the built FP-Tree
pub fn mine_patterns(state: &StreamingState, min_support: f64) -> Result<Vec<FrequentLevel>, String> {
    if state.phase != ProcessingPhase::ReadyToMine {
        return Err(format!("Cannot mine in phase {:?}", state.phase));
    }

    let fp_tree = state.fp_tree.as_ref()
        .ok_or("FP-Tree not initialized")?;

    let result = fp_growth_recursive(
        fp_tree,
        &state.frequent_items,
        &[],
        min_support,
        state.num_transactions
    );

    Ok(result)
}

/// Recursive FP-Growth mining (same as regular implementation)
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
        let mut pattern = alpha.to_vec();
        pattern.extend_from_slice(current);

        let size = pattern.len();
        while result.len() < size {
            result.push(FrequentLevel::new(result.len() + 1));
        }
        result[size - 1].add_itemset(pattern);
        return;
    }

    for i in start..path.len() {
        current.push(path[i].0);
        generate_comb_recursive(path, k, i + 1, current, alpha, result);
        current.pop();
    }
}
