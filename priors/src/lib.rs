use std::collections::HashMap;
use numpy::ndarray::{Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use rayon::prelude::*;

/// Memory-efficient itemset storage using flat arrays
#[derive(Debug, Clone)]
struct ItemsetStorage {
    /// Flat storage for all items across all itemsets
    items: Vec<usize>,
    /// Offsets into the items array: (start_idx, length) for each itemset
    offsets: Vec<(usize, usize)>,
}

impl ItemsetStorage {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn with_capacity(estimated_items: usize, estimated_itemsets: usize) -> Self {
        Self {
            items: Vec::with_capacity(estimated_items),
            offsets: Vec::with_capacity(estimated_itemsets),
        }
    }

    /// Add an itemset to storage, returns the index
    fn add_itemset(&mut self, mut items: Vec<usize>) -> usize {
        items.sort_unstable();
        items.dedup();

        let start_idx = self.items.len();
        let length = items.len();

        self.items.extend_from_slice(&items);
        self.offsets.push((start_idx, length));

        self.offsets.len() - 1
    }

    /// Get an itemset by index
    fn get_itemset(&self, idx: usize) -> &[usize] {
        let (start, len) = self.offsets[idx];
        &self.items[start..start + len]
    }

    /// Get the number of itemsets
    fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Get itemset length
    fn itemset_len(&self, idx: usize) -> usize {
        self.offsets[idx].1
    }
}

/// Memory-efficient level storage for frequent itemsets
#[derive(Debug, Clone)]
struct FrequentLevel {
    storage: ItemsetStorage,
    itemset_size: usize,
}

impl FrequentLevel {
    fn new(itemset_size: usize) -> Self {
        Self {
            storage: ItemsetStorage::new(),
            itemset_size,
        }
    }

    fn with_capacity(itemset_size: usize, estimated_itemsets: usize) -> Self {
        let estimated_items = estimated_itemsets * itemset_size;
        Self {
            storage: ItemsetStorage::with_capacity(estimated_items, estimated_itemsets),
            itemset_size,
        }
    }

    fn add_itemset(&mut self, items: Vec<usize>) -> usize {
        debug_assert_eq!(items.len(), self.itemset_size);
        self.storage.add_itemset(items)
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn get_itemset(&self, idx: usize) -> &[usize] {
        self.storage.get_itemset(idx)
    }

    fn iter_itemsets(&self) -> impl Iterator<Item = &[usize]> {
        (0..self.len()).map(move |i| self.get_itemset(i))
    }
}

// ============================================================================
// FP-GROWTH ALGORITHM IMPLEMENTATION
// ============================================================================

/// FP-Tree node for efficient pattern mining
#[derive(Debug, Clone)]
struct FPNode {
    item: Option<usize>,
    count: usize,
    parent: Option<usize>,  // Index into nodes vector
    children: HashMap<usize, usize>,  // item -> node_index
}

impl FPNode {
    fn new_root() -> Self {
        Self {
            item: None,
            count: 0,
            parent: None,
            children: HashMap::new(),
        }
    }

    fn new_item(item: usize, count: usize, parent: Option<usize>) -> Self {
        Self {
            item: Some(item),
            count,
            parent,
            children: HashMap::new(),
        }
    }
}

/// FP-Tree data structure for compact transaction storage
#[derive(Debug, Clone)]
struct FPTree {
    nodes: Vec<FPNode>,
    header_table: HashMap<usize, Vec<usize>>,  // item -> list of node indices
    root_index: usize,
}

impl FPTree {
    fn new() -> Self {
        let mut nodes = Vec::new();
        let root = FPNode::new_root();
        nodes.push(root);

        Self {
            nodes,
            header_table: HashMap::new(),
            root_index: 0,
        }
    }

    /// Insert a transaction into the FP-Tree
    fn insert_transaction(&mut self, transaction: &[usize], counts: &[usize]) {
        let mut current_index = self.root_index;

        for (&item, &count) in transaction.iter().zip(counts.iter()) {
            let current_node = &self.nodes[current_index];

            if let Some(&child_index) = current_node.children.get(&item) {
                // Item exists as child, increment count
                self.nodes[child_index].count += count;
                current_index = child_index;
            } else {
                // Create new child node
                let new_node = FPNode::new_item(item, count, Some(current_index));
                let new_index = self.nodes.len();
                self.nodes.push(new_node);

                // Update parent's children
                self.nodes[current_index].children.insert(item, new_index);

                // Update header table
                self.header_table.entry(item)
                    .or_insert_with(Vec::new)
                    .push(new_index);

                current_index = new_index;
            }
        }
    }

    /// Get all paths ending with the given item
    fn get_prefix_paths(&self, item: usize) -> Vec<(Vec<usize>, usize)> {
        let mut prefix_paths = Vec::new();

        if let Some(node_indices) = self.header_table.get(&item) {
            for &node_index in node_indices {
                let node = &self.nodes[node_index];
                let count = node.count;

                // Trace back to root to get prefix path
                let mut path = Vec::new();
                let mut current_index = node.parent;

                while let Some(parent_index) = current_index {
                    let parent_node = &self.nodes[parent_index];
                    if let Some(parent_item) = parent_node.item {
                        path.push(parent_item);
                    }
                    current_index = parent_node.parent;
                }

                path.reverse();
                if !path.is_empty() {
                    prefix_paths.push((path, count));
                }
            }
        }

        prefix_paths
    }

    /// Check if tree has single path (optimization for FP-Growth)
    fn has_single_path(&self) -> bool {
        let mut current_index = self.root_index;

        loop {
            let current_node = &self.nodes[current_index];

            if current_node.children.len() > 1 {
                return false; // Multiple branches
            }

            if current_node.children.is_empty() {
                return true; // Reached leaf
            }

            // Move to only child
            current_index = *current_node.children.values().next().unwrap();
        }
    }

    /// Get single path items (for optimization)
    fn get_single_path(&self) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        let mut current_index = self.root_index;

        loop {
            let current_node = &self.nodes[current_index];

            if current_node.children.is_empty() {
                break;
            }

            // Move to only child
            current_index = *current_node.children.values().next().unwrap();
            let child_node = &self.nodes[current_index];

            if let Some(item) = child_node.item {
                path.push((item, child_node.count));
            }
        }

        path
    }
}

/// Build FP-Tree from transactions with minimum support filtering
fn build_fp_tree(transactions: ArrayView2<i32>, min_support: f64) -> (FPTree, Vec<usize>) {
    let num_transactions = transactions.shape()[0];
    let min_count = (min_support * num_transactions as f64) as usize;

    // Step 1: Count item frequencies
    let item_counts = transactions.sum_axis(Axis(0));

    // Step 2: Filter frequent items and sort by frequency (descending)
    let mut frequent_items: Vec<(usize, usize)> = item_counts
        .iter()
        .enumerate()
        .filter_map(|(item, &count)| {
            if count as usize >= min_count {
                Some((item, count as usize))
            } else {
                None
            }
        })
        .collect();

    // Sort by frequency (descending) - this is crucial for FP-Tree efficiency
    frequent_items.sort_by(|a, b| b.1.cmp(&a.1));

    let _frequent_item_set: HashMap<usize, usize> = frequent_items.iter()
        .enumerate()
        .map(|(rank, &(item, _))| (item, rank))
        .collect();

    let ordered_items: Vec<usize> = frequent_items.iter().map(|&(item, _)| item).collect();

    // Step 3: Build FP-Tree
    let mut fp_tree = FPTree::new();

    for tx_idx in 0..num_transactions {
        // Extract frequent items from this transaction, ordered by global frequency
        let mut tx_items: Vec<usize> = Vec::new();

        for &item in &ordered_items {
            if transactions[[tx_idx, item]] != 0 {
                tx_items.push(item);
            }
        }

        if !tx_items.is_empty() {
            let counts = vec![1; tx_items.len()]; // Each item appears once per transaction
            fp_tree.insert_transaction(&tx_items, &counts);
        }
    }

    (fp_tree, ordered_items)
}

/// Parallel FP-Growth recursive function with collect-and-merge approach
fn fp_growth_recursive_parallel(
    fp_tree: &FPTree,
    frequent_items: &[usize],
    alpha: &[usize],
    min_support: f64,
    num_transactions: usize,
) -> Vec<FrequentLevel> {
    let min_count = (min_support * num_transactions as f64) as usize;
    let mut local_result = Vec::new();

    // Check if tree has single path (optimization)
    if fp_tree.has_single_path() {
        let path = fp_tree.get_single_path();
        let mut alpha_vec = alpha.to_vec();

        // Generate all combinations of the path
        for i in 1..=path.len() {
            generate_combinations_from_path(&path, i, &mut alpha_vec, &mut local_result);
        }
        return local_result;
    }

    // Process items in parallel with collect-and-merge
    let parallel_results: Vec<Vec<FrequentLevel>> = frequent_items
        .par_iter()
        .rev()  // Process in reverse frequency order
        .map(|&item| {
            let mut item_result = Vec::new();

            // Create new pattern by adding item to alpha
            let mut new_pattern = alpha.to_vec();
            new_pattern.push(item);

            // Get item support from header table
            let item_support = if let Some(node_indices) = fp_tree.header_table.get(&item) {
                node_indices.iter().map(|&idx| fp_tree.nodes[idx].count).sum::<usize>()
            } else {
                0
            };

            if item_support >= min_count {
                // Add current pattern as frequent
                add_pattern_to_result(&new_pattern, &mut item_result);

                // Get prefix paths for this item
                let prefix_paths = fp_tree.get_prefix_paths(item);

                if !prefix_paths.is_empty() {
                    // Build conditional FP-Tree
                    let conditional_tree = build_conditional_fp_tree(&prefix_paths, min_count);

                    // Get frequent items for conditional tree
                    let conditional_frequent_items = get_conditional_frequent_items(&conditional_tree, min_count);

                    if !conditional_frequent_items.is_empty() {
                        // Recursive call (sequential within each parallel branch)
                        let recursive_results = fp_growth_recursive_parallel(
                            &conditional_tree,
                            &conditional_frequent_items,
                            &new_pattern,
                            min_support,
                            num_transactions,
                        );

                        // Merge recursive results
                        item_result.extend(recursive_results);
                    }
                }
            }

            item_result
        })
        .collect();

    // Merge all parallel results
    for item_results in parallel_results {
        local_result.extend(item_results);
    }

    local_result
}

/// Helper function to add a pattern to the result
fn add_pattern_to_result(pattern: &[usize], result: &mut Vec<FrequentLevel>) {
    let pattern_size = pattern.len();

    // Ensure we have enough levels
    while result.len() < pattern_size {
        result.push(FrequentLevel::new(result.len() + 1));
    }

    // Add pattern to appropriate level
    if pattern_size > 0 {
        result[pattern_size - 1].add_itemset(pattern.to_vec());
    }
}

/// Build conditional FP-Tree from prefix paths
fn build_conditional_fp_tree(prefix_paths: &[(Vec<usize>, usize)], min_count: usize) -> FPTree {
    let mut item_counts: HashMap<usize, usize> = HashMap::new();

    // Count items in prefix paths
    for (path, count) in prefix_paths {
        for &item in path {
            *item_counts.entry(item).or_insert(0) += count;
        }
    }

    // Filter frequent items
    let frequent_items: Vec<usize> = item_counts.iter()
        .filter_map(|(&item, &count)| if count >= min_count { Some(item) } else { None })
        .collect();

    // Build conditional tree
    let mut conditional_tree = FPTree::new();

    for (path, count) in prefix_paths {
        let filtered_path: Vec<usize> = path.iter()
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

/// Get frequent items from conditional tree
fn get_conditional_frequent_items(tree: &FPTree, min_count: usize) -> Vec<usize> {
    let mut item_counts: HashMap<usize, usize> = HashMap::new();

    for (&item, node_indices) in &tree.header_table {
        let total_count: usize = node_indices.iter()
            .map(|&idx| tree.nodes[idx].count)
            .sum();
        item_counts.insert(item, total_count);
    }

    let mut frequent_items: Vec<(usize, usize)> = item_counts.iter()
        .filter_map(|(&item, &count)| if count >= min_count { Some((item, count)) } else { None })
        .collect();

    // Sort by frequency (descending)
    frequent_items.sort_by(|a, b| b.1.cmp(&a.1));

    frequent_items.into_iter().map(|(item, _)| item).collect()
}

/// Generate combinations from single path
fn generate_combinations_from_path(
    path: &[(usize, usize)],
    k: usize,
    alpha: &[usize],
    result: &mut Vec<FrequentLevel>
) {
    if k == 0 || k > path.len() { return; }

    // Generate all k-combinations
    let indices: Vec<usize> = (0..path.len()).collect();
    let mut callback = |combination: &[usize]| {
        let mut pattern = alpha.to_vec();
        for &idx in combination {
            pattern.push(path[idx].0);
        }
        add_pattern_to_result(&pattern, result);
    };
    generate_combinations_recursive(&indices, k, 0, &mut Vec::new(), &mut callback);
}

/// Recursive combination generation helper
fn generate_combinations_recursive<F>(
    items: &[usize],
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    callback: &mut F
) where F: FnMut(&[usize]) {
    if current.len() == k {
        callback(current);
        return;
    }

    for i in start..items.len() {
        current.push(items[i]);
        generate_combinations_recursive(items, k, i + 1, current, callback);
        current.pop();
    }
}

/// Main FP-Growth algorithm entry point
fn fp_growth_algorithm(transactions: ArrayView2<i32>, min_support: f64) -> Vec<FrequentLevel> {
    let num_transactions = transactions.shape()[0];

    // Build FP-Tree
    let (fp_tree, frequent_items) = build_fp_tree(transactions, min_support);

    // Use parallel FP-Growth with collect-and-merge
    let alpha = Vec::new();
    fp_growth_recursive_parallel(&fp_tree, &frequent_items, &alpha, min_support, num_transactions)
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

#[pymodule]
fn priors<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {

    /// FP-Growth algorithm Python wrapper
    ///
    /// Parameters:
    /// - transactions: 2D binary matrix where rows are transactions and columns are items
    /// - min_support: minimum support threshold (between 0 and 1)
    ///
    /// Returns:
    /// - List of frequent itemsets (as lists of item indices) for each level
    #[pyfn(m)]
    #[pyo3(name = "fp_growth")]
    fn fp_growth_py<'py>(
        py: Python<'py>,
        transactions: PyReadonlyArray2<'py, i32>,
        min_support: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray2<usize>>>> {
        let transactions_view = transactions.as_array();
        let frequent_levels = fp_growth_algorithm(transactions_view, min_support);

        let mut result = Vec::new();

        for level in frequent_levels {
            if level.len() == 0 {
                continue;
            }

            let itemset_size = level.itemset_size;
            let num_itemsets = level.len();

            // Create a 2D array: rows = itemsets, cols = items in each itemset
            let mut data = vec![0usize; num_itemsets * itemset_size];

            for (i, itemset) in level.iter_itemsets().enumerate() {
                for (j, &item) in itemset.iter().enumerate() {
                    data[i * itemset_size + j] = item;
                }
            }

            let array = Array2::from_shape_vec((num_itemsets, itemset_size), data)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to create array"))?;

            result.push(array.into_pyarray(py));
        }

        Ok(result)
    }

    Ok(())
}
