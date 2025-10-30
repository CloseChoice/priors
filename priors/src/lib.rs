use std::collections::HashMap;
use numpy::ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use rayon::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;

mod lazy_fp_growth;
use lazy_fp_growth::LazyFPGrowth;

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

/// SIMD-optimized support calculation using vectorized operations and flat storage
fn calculate_support_flat(transactions: ArrayView2<i32>, level: &FrequentLevel) -> Vec<f64> {
    let num_transactions = transactions.shape()[0] as f64;
    let mut support_values = Vec::with_capacity(level.len());
    
    for itemset in level.iter_itemsets() {
        if itemset.is_empty() {
            support_values.push(0.0);
            continue;
        }
        
        // Start with the first item's column
        let mut mask = transactions.column(itemset[0]).to_owned();
        
        // Vectorized AND operation across all items in the itemset
        for &item in &itemset[1..] {
            let item_column = transactions.column(item);
            // Element-wise AND using ndarray's zip
            mask.zip_mut_with(&item_column, |a, &b| *a = (*a != 0 && b != 0) as i32);
        }
        
        // Vectorized sum to count transactions containing all items
        let count = mask.sum() as f64;
        let support = count / num_transactions;
        support_values.push(support);
    }
    
    support_values
}

/// Fast frequent 1-itemsets generation using vectorized operations
fn generate_frequent_1_itemsets_flat(transactions: ArrayView2<i32>, min_support: f64) -> FrequentLevel {
    let num_transactions = transactions.shape()[0] as f64;
    let num_items = transactions.shape()[1];
    
    // Pre-allocate for worst case (all items frequent)
    let mut level = FrequentLevel::with_capacity(1, num_items);
    
    // Vectorized sum across transactions for each item (column-wise sum)
    let item_counts = transactions.sum_axis(Axis(0));
    
    for (item_idx, &count) in item_counts.iter().enumerate() {
        let support = count as f64 / num_transactions;
        if support >= min_support {
            level.add_itemset(vec![item_idx]);
        }
    }
    
    level
}

/// Optimized candidate generation using flat storage
fn generate_candidates_flat(frequent_level: &FrequentLevel) -> FrequentLevel {
    let k = frequent_level.itemset_size;
    let new_k = k + 1;
    
    // Estimate number of candidates (conservative upper bound)
    let num_frequent = frequent_level.len();
    let estimated_candidates = if num_frequent < 2 { 0 } else { num_frequent * (num_frequent - 1) / 2 };
    
    let mut candidates = FrequentLevel::with_capacity(new_k, estimated_candidates);
    
    // Generate candidates by joining frequent itemsets
    for i in 0..num_frequent {
        let itemset1 = frequent_level.get_itemset(i);
        
        for j in (i + 1)..num_frequent {
            let itemset2 = frequent_level.get_itemset(j);
            
            // Check if first k-1 items are the same (for lexicographic join)
            if k == 0 || (k > 0 && itemset1[..k-1] == itemset2[..k-1]) {
                // Create new candidate by merging
                let mut new_items = Vec::with_capacity(new_k);
                new_items.extend_from_slice(itemset1);
                new_items.extend_from_slice(itemset2);
                new_items.sort_unstable();
                new_items.dedup();
                
                // Only add if it's actually a (k+1)-itemset
                if new_items.len() == new_k {
                    candidates.add_itemset(new_items);
                }
            }
        }
    }
    
    candidates
}

/// Optimized pruning using flat storage and better algorithms
fn prune_candidates_flat(candidates: FrequentLevel, frequent_level: &FrequentLevel) -> FrequentLevel {
    if candidates.len() == 0 || frequent_level.len() == 0 {
        return candidates;
    }
    
    let k = candidates.itemset_size;
    if k <= 1 {
        return candidates; // No pruning needed for 1-itemsets
    }
    
    // Build a faster lookup structure - use sorted vectors instead of HashSet
    let mut frequent_itemsets: Vec<Vec<usize>> = Vec::with_capacity(frequent_level.len());
    for itemset in frequent_level.iter_itemsets() {
        frequent_itemsets.push(itemset.to_vec());
    }
    frequent_itemsets.sort_unstable(); // Sort for binary search
    
    let mut pruned = FrequentLevel::with_capacity(k, candidates.len());
    
    'candidate_loop: for candidate in candidates.iter_itemsets() {
        // Check if all (k-1)-subsets are frequent using binary search
        for i in 0..candidate.len() {
            let mut subset = candidate.to_vec();
            subset.remove(i);
            
            // Binary search in sorted frequent itemsets
            if frequent_itemsets.binary_search(&subset).is_err() {
                continue 'candidate_loop; // This candidate should be pruned
            }
        }
        
        // All subsets are frequent, keep this candidate
        pruned.add_itemset(candidate.to_vec());
    }
    
    pruned
}

/// Main memory-optimized Apriori algorithm
fn apriori_algorithm_flat(transactions: ArrayView2<i32>, min_support: f64) -> Vec<FrequentLevel> {
    let mut all_frequent_levels = Vec::new();
    
    // Generate frequent 1-itemsets using vectorized operations
    let mut frequent_k = generate_frequent_1_itemsets_flat(transactions, min_support);
    
    while frequent_k.len() > 0 {
        all_frequent_levels.push(frequent_k.clone());
        
        // Generate candidates more efficiently
        let candidates = generate_candidates_flat(&frequent_k);
        
        if candidates.len() == 0 {
            break;
        }
        
        // Prune candidates using optimized lookup
        let pruned_candidates = prune_candidates_flat(candidates, &frequent_k);
        
        if pruned_candidates.len() == 0 {
            break;
        }
        
        // Calculate support using vectorized operations
        let support_values = calculate_support_flat(transactions, &pruned_candidates);
        
        // Filter by minimum support and build next level
        let next_k = pruned_candidates.itemset_size;
        let mut next_level = FrequentLevel::with_capacity(next_k, pruned_candidates.len());
        
        for (idx, &support) in support_values.iter().enumerate() {
            if support >= min_support {
                let itemset = pruned_candidates.get_itemset(idx).to_vec();
                next_level.add_itemset(itemset);
            }
        }
        
        frequent_k = next_level;
    }
    
    all_frequent_levels
}

/// Optimized confidence calculation using flat storage
fn calculate_confidence_flat(transactions: ArrayView2<i32>, antecedent: &[usize], consequent: &[usize]) -> f64 {
    // Create temporary levels for support calculation
    let mut antecedent_level = FrequentLevel::new(antecedent.len());
    antecedent_level.add_itemset(antecedent.to_vec());
    
    let antecedent_support = calculate_support_flat(transactions, &antecedent_level)[0];
    
    if antecedent_support == 0.0 {
        return 0.0;
    }
    
    // Calculate combined support
    let mut combined_items = antecedent.to_vec();
    combined_items.extend_from_slice(consequent);
    combined_items.sort_unstable();
    combined_items.dedup();
    
    let mut combined_level = FrequentLevel::new(combined_items.len());
    combined_level.add_itemset(combined_items);
    
    let combined_support = calculate_support_flat(transactions, &combined_level)[0];
    
    combined_support / antecedent_support
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

/// FP-Growth recursive mining algorithm
fn fp_growth_recursive(
    fp_tree: &FPTree, 
    frequent_items: &[usize],
    alpha: &mut Vec<usize>,
    min_support: f64,
    num_transactions: usize,
    result: &mut Vec<FrequentLevel>
) {
    let min_count = (min_support * num_transactions as f64) as usize;
    
    // Check if tree has single path (optimization)
    if fp_tree.has_single_path() {
        let path = fp_tree.get_single_path();
        
        // Generate all combinations of the path
        for i in 1..=path.len() {
            generate_combinations_from_path(&path, i, alpha, result);
        }
        return;
    }
    
    // Process items in reverse frequency order (least frequent first)
    for &item in frequent_items.iter().rev() {
        // Create new pattern by adding item to alpha
        let mut new_pattern = alpha.clone();
        new_pattern.push(item);
        
        // Get item support from header table
        let item_support = if let Some(node_indices) = fp_tree.header_table.get(&item) {
            node_indices.iter().map(|&idx| fp_tree.nodes[idx].count).sum::<usize>()
        } else {
            0
        };
        
        if item_support >= min_count {
            // Add frequent pattern
            add_pattern_to_result(&new_pattern, result);
            
            // Get prefix paths for conditional FP-Tree
            let prefix_paths = fp_tree.get_prefix_paths(item);
            
            if !prefix_paths.is_empty() {
                // Build conditional FP-Tree
                let conditional_tree = build_conditional_fp_tree(&prefix_paths, min_count);
                
                // Get frequent items for conditional tree
                let conditional_frequent_items = get_conditional_frequent_items(&conditional_tree, min_count);
                
                if !conditional_frequent_items.is_empty() {
                    // Recursive call
                    fp_growth_recursive(
                        &conditional_tree,
                        &conditional_frequent_items,
                        &mut new_pattern,
                        min_support,
                        num_transactions,
                        result
                    );
                }
            }
        }
    }
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
// BIT VECTOR OPTIMIZED ALGORITHMS (for datasets with ≤64 items)
// ============================================================================

/// Convert binary transaction matrix to bit vector representation
fn convert_to_bitvectors(transactions: ArrayView2<i32>) -> Vec<u64> {
    let num_transactions = transactions.shape()[0];
    let num_items = transactions.shape()[1];
    
    if num_items > 64 {
        panic!("Bit vector optimization only supports ≤64 items, got {}", num_items);
    }
    
    let mut bitvectors = Vec::with_capacity(num_transactions);
    
    for tx_idx in 0..num_transactions {
        let mut bitvector = 0u64;
        
        for item_idx in 0..num_items {
            if transactions[[tx_idx, item_idx]] != 0 {
                bitvector |= 1u64 << item_idx;
            }
        }
        
        bitvectors.push(bitvector);
    }
    
    bitvectors
}

/// Ultra-fast support calculation using bit operations
fn calculate_support_bitvector(transactions: &[u64], itemset: u64) -> f64 {
    let total_transactions = transactions.len() as f64;
    
    if total_transactions == 0.0 {
        return 0.0;
    }
    
    // Count transactions that contain all items in the itemset
    let count = transactions.par_iter()
        .map(|&transaction| {
            // Check if all items in itemset are present in transaction
            ((transaction & itemset) == itemset) as usize
        })
        .sum::<usize>();
    
    count as f64 / total_transactions
}

/// Generate frequent 1-itemsets using bit vectors
fn generate_frequent_1_itemsets_bitvector(transactions: &[u64], num_items: usize, min_support: f64) -> Vec<u64> {
    let mut frequent_items = Vec::new();
    
    // Check each individual item
    for item_idx in 0..num_items {
        let itemset = 1u64 << item_idx;
        let support = calculate_support_bitvector(transactions, itemset);
        
        if support >= min_support {
            frequent_items.push(itemset);
        }
    }
    
    frequent_items
}

/// Generate candidate itemsets by combining frequent itemsets
fn generate_candidates_bitvector(frequent_itemsets: &[u64]) -> Vec<u64> {
    let mut candidates = Vec::new();
    
    // Generate candidates by combining pairs of frequent itemsets
    for i in 0..frequent_itemsets.len() {
        for j in (i + 1)..frequent_itemsets.len() {
            let itemset1 = frequent_itemsets[i];
            let itemset2 = frequent_itemsets[j];
            
            // Combine itemsets using bitwise OR
            let candidate = itemset1 | itemset2;
            
            // Only add if the candidate has exactly one more item than the originals
            let itemset1_count = itemset1.count_ones();
            let itemset2_count = itemset2.count_ones();
            let candidate_count = candidate.count_ones();
            
            // For proper candidate generation, we want k-itemsets to generate (k+1)-itemsets
            if itemset1_count == itemset2_count && candidate_count == itemset1_count + 1 {
                // Check if this candidate hasn't been added already
                if !candidates.contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }
    }
    
    candidates
}

/// Prune candidates using the Apriori property (optimized for bit vectors)
fn prune_candidates_bitvector(candidates: Vec<u64>, frequent_itemsets: &[u64]) -> Vec<u64> {
    if candidates.is_empty() || frequent_itemsets.is_empty() {
        return candidates;
    }
    
    // Convert frequent itemsets to HashSet for O(1) lookup
    let frequent_set: std::collections::HashSet<u64> = frequent_itemsets.iter().cloned().collect();
    
    candidates.into_par_iter()
        .filter(|&candidate| {
            let candidate_count = candidate.count_ones();
            
            if candidate_count <= 1 {
                return true; // No pruning needed for 1-itemsets
            }
            
            // Check if all (k-1)-subsets are frequent
            for bit_pos in 0..64 {
                if (candidate & (1u64 << bit_pos)) != 0 {
                    // Create subset by removing this item
                    let subset = candidate & !(1u64 << bit_pos);
                    
                    if !frequent_set.contains(&subset) {
                        return false; // This candidate should be pruned
                    }
                }
            }
            
            true
        })
        .collect()
}

/// Convert bit vector itemsets back to our FrequentLevel format
fn convert_bitvectors_to_levels(bitvector_levels: Vec<Vec<u64>>) -> Vec<FrequentLevel> {
    let mut levels = Vec::new();
    
    for (level_idx, bitvector_itemsets) in bitvector_levels.into_iter().enumerate() {
        let itemset_size = level_idx + 1;
        let mut level = FrequentLevel::with_capacity(itemset_size, bitvector_itemsets.len());
        
        for bitvector in bitvector_itemsets {
            // Convert bitvector back to item list
            let mut items = Vec::new();
            for bit_pos in 0..64 {
                if (bitvector & (1u64 << bit_pos)) != 0 {
                    items.push(bit_pos);
                }
            }
            
            if items.len() == itemset_size {
                level.add_itemset(items);
            }
        }
        
        levels.push(level);
    }
    
    levels
}

/// Ultra-fast bit vector Apriori algorithm
fn apriori_bitvector(transactions: ArrayView2<i32>, min_support: f64) -> Vec<FrequentLevel> {
    let num_items = transactions.shape()[1];
    
    if num_items > 64 {
        // Fall back to regular algorithm for large item sets
        return apriori_algorithm_flat(transactions, min_support);
    }
    
    // Convert to bit vectors
    let bitvector_transactions = convert_to_bitvectors(transactions);
    
    let mut all_frequent_levels = Vec::new();
    
    // Generate frequent 1-itemsets
    let mut frequent_k = generate_frequent_1_itemsets_bitvector(&bitvector_transactions, num_items, min_support);
    
    while !frequent_k.is_empty() {
        all_frequent_levels.push(frequent_k.clone());
        
        // Generate candidates
        let candidates = generate_candidates_bitvector(&frequent_k);
        
        if candidates.is_empty() {
            break;
        }
        
        // Prune candidates
        let pruned_candidates = prune_candidates_bitvector(candidates, &frequent_k);
        
        if pruned_candidates.is_empty() {
            break;
        }
        
        // Calculate support for pruned candidates (in parallel)
        frequent_k = pruned_candidates.into_par_iter()
            .filter_map(|candidate| {
                let support = calculate_support_bitvector(&bitvector_transactions, candidate);
                if support >= min_support {
                    Some(candidate)
                } else {
                    None
                }
            })
            .collect();
    }
    
    // Convert back to FrequentLevel format
    convert_bitvectors_to_levels(all_frequent_levels)
}

/// Ultra-fast bit vector FP-Growth (simplified version focusing on speed)
fn fp_growth_bitvector(transactions: ArrayView2<i32>, min_support: f64) -> Vec<FrequentLevel> {
    let num_items = transactions.shape()[1];
    
    if num_items > 64 {
        // Fall back to regular FP-Growth for large item sets
        return fp_growth_algorithm(transactions, min_support);
    }
    
    // For now, use bit vector Apriori as it's simpler to implement correctly
    // FP-Growth with bit vectors requires more complex tree operations
    apriori_bitvector(transactions, min_support)
}

// ============================================================================
// DERAR ALGORITHM IMPLEMENTATION (Dynamic Extracting of Relevant Association Rules)
// ============================================================================

/// DERAR Rule with statistical quality measures
#[derive(Debug, Clone)]
struct DERARule {
    antecedent: Vec<usize>,
    consequent: Vec<usize>,
    support: f64,
    confidence: f64,
    mutual_information: f64,
    stability_score: f64,
    target_concentration_measure: f64,
}

impl DERARule {
    fn new(antecedent: Vec<usize>, consequent: Vec<usize>) -> Self {
        Self {
            antecedent,
            consequent,
            support: 0.0,
            confidence: 0.0,
            mutual_information: 0.0,
            stability_score: 0.0,
            target_concentration_measure: 0.0,
        }
    }
}

/// Meta-Pattern Tree Node (enhanced FP-Tree for DERAR)
#[derive(Debug, Clone)]
struct MetaPatternNode {
    item: Option<usize>,
    count: usize,
    parent: Option<usize>,
    children: HashMap<usize, usize>,
    // DERAR-specific enhancements
    stability_score: f64,
    pattern_quality: f64,
}

impl MetaPatternNode {
    fn new_root() -> Self {
        Self {
            item: None,
            count: 0,
            parent: None,
            children: HashMap::new(),
            stability_score: 0.0,
            pattern_quality: 0.0,
        }
    }
    
    fn new_item(item: usize, count: usize, parent: Option<usize>) -> Self {
        Self {
            item: Some(item),
            count,
            parent,
            children: HashMap::new(),
            stability_score: 0.0,
            pattern_quality: 0.0,
        }
    }
}

/// Meta-Pattern Tree for DERAR algorithm
#[derive(Debug, Clone)]
struct MetaPatternTree {
    nodes: Vec<MetaPatternNode>,
    header_table: HashMap<usize, Vec<usize>>,
    root_index: usize,
    // Statistical tracking
    item_cooccurrence: HashMap<(usize, usize), usize>,
    total_transactions: usize,
}

impl MetaPatternTree {
    fn new() -> Self {
        let mut nodes = Vec::new();
        let root = MetaPatternNode::new_root();
        nodes.push(root);
        
        Self {
            nodes,
            header_table: HashMap::new(),
            root_index: 0,
            item_cooccurrence: HashMap::new(),
            total_transactions: 0,
        }
    }
    
    /// Insert transaction with statistical tracking
    fn insert_transaction_with_stats(&mut self, transaction: &[usize], counts: &[usize]) {
        self.total_transactions += 1;
        
        // Track co-occurrence for mutual information calculation
        for i in 0..transaction.len() {
            for j in (i + 1)..transaction.len() {
                let pair = if transaction[i] < transaction[j] {
                    (transaction[i], transaction[j])
                } else {
                    (transaction[j], transaction[i])
                };
                *self.item_cooccurrence.entry(pair).or_insert(0) += 1;
            }
        }
        
        // Insert into tree (similar to FP-Tree)
        let mut current_index = self.root_index;
        
        for (&item, &count) in transaction.iter().zip(counts.iter()) {
            let current_node = &self.nodes[current_index];
            
            if let Some(&child_index) = current_node.children.get(&item) {
                self.nodes[child_index].count += count;
                current_index = child_index;
            } else {
                let new_node = MetaPatternNode::new_item(item, count, Some(current_index));
                let new_index = self.nodes.len();
                self.nodes.push(new_node);
                
                self.nodes[current_index].children.insert(item, new_index);
                
                self.header_table.entry(item)
                    .or_insert_with(Vec::new)
                    .push(new_index);
                
                current_index = new_index;
            }
        }
    }
}

/// Calculate Mutual Information between two items
fn calculate_mutual_information(
    tree: &MetaPatternTree,
    item_a: usize,
    item_b: usize,
    item_a_count: usize,
    item_b_count: usize,
) -> f64 {
    let total = tree.total_transactions as f64;
    
    if total == 0.0 {
        return 0.0;
    }
    
    // P(A) and P(B)
    let p_a = item_a_count as f64 / total;
    let p_b = item_b_count as f64 / total;
    
    // P(A,B) - probability of co-occurrence
    let pair = if item_a < item_b { (item_a, item_b) } else { (item_b, item_a) };
    let cooccurrence_count = tree.item_cooccurrence.get(&pair).cloned().unwrap_or(0);
    let p_ab = cooccurrence_count as f64 / total;
    
    // Mutual Information: MI(A,B) = P(A,B) * log(P(A,B) / (P(A) * P(B)))
    if p_ab > 0.0 && p_a > 0.0 && p_b > 0.0 {
        p_ab * (p_ab / (p_a * p_b)).ln()
    } else {
        0.0
    }
}

/// Calculate Stability Score (percentage of transactions containing the pattern)
fn calculate_stability_score(pattern_support: f64, total_transactions: usize) -> f64 {
    if total_transactions == 0 {
        return 0.0;
    }
    pattern_support * 100.0 // Convert to percentage
}

/// Calculate Target Concentration Measure (TCM) for semantic quality
fn calculate_tcm(
    antecedent: &[usize],
    consequent: &[usize],
    tree: &MetaPatternTree,
    transactions: ArrayView2<i32>,
) -> f64 {
    // TCM measures how "concentrated" the consequent is given the antecedent
    // Higher TCM means the rule is more semantically meaningful
    
    let num_transactions = transactions.shape()[0];
    
    // Count transactions with antecedent
    let antecedent_count = (0..num_transactions)
        .map(|tx_idx| {
            let has_antecedent = antecedent.iter()
                .all(|&item| transactions[[tx_idx, item]] != 0);
            has_antecedent as usize
        })
        .sum::<usize>();
    
    if antecedent_count == 0 {
        return 0.0;
    }
    
    // Count transactions with both antecedent and consequent
    let both_count = (0..num_transactions)
        .map(|tx_idx| {
            let has_antecedent = antecedent.iter()
                .all(|&item| transactions[[tx_idx, item]] != 0);
            let has_consequent = consequent.iter()
                .all(|&item| transactions[[tx_idx, item]] != 0);
            (has_antecedent && has_consequent) as usize
        })
        .sum::<usize>();
    
    // TCM = concentration ratio
    both_count as f64 / antecedent_count as f64
}

/// DERAR's intelligent rule filtering based on statistical measures
fn filter_rules_derar(
    rules: Vec<DERARule>,
    stability_threshold: f64,
    mi_threshold: f64,
    confidence_threshold: f64,
    tcm_threshold: f64,
) -> Vec<DERARule> {
    rules.into_iter()
        .filter(|rule| {
            // Multi-stage filtering as per DERAR paper
            rule.stability_score >= stability_threshold &&
            rule.mutual_information >= mi_threshold &&
            rule.confidence >= confidence_threshold &&
            rule.target_concentration_measure >= tcm_threshold
        })
        .collect()
}

/// Generate association rules with DERAR quality measures
fn generate_derar_rules(
    frequent_patterns: &[Vec<usize>],
    tree: &MetaPatternTree,
    transactions: ArrayView2<i32>,
    min_confidence: f64,
) -> Vec<DERARule> {
    let mut rules = Vec::new();
    
    // Generate rules from frequent patterns (2+ items)
    for pattern in frequent_patterns {
        if pattern.len() < 2 {
            continue;
        }
        
        // Generate all possible antecedent/consequent splits
        for i in 1..pattern.len() {
            for antecedent_mask in 0u32..(1u32 << pattern.len()) {
                if antecedent_mask.count_ones() as usize != i {
                    continue;
                }
                
                let mut antecedent = Vec::new();
                let mut consequent = Vec::new();
                
                for (bit_pos, &item) in pattern.iter().enumerate() {
                    if (antecedent_mask & (1 << bit_pos)) != 0 {
                        antecedent.push(item);
                    } else {
                        consequent.push(item);
                    }
                }
                
                if antecedent.is_empty() || consequent.is_empty() {
                    continue;
                }
                
                // Calculate basic measures
                let support = calculate_pattern_support(&pattern, transactions);
                let antecedent_support = calculate_pattern_support(&antecedent, transactions);
                let confidence = if antecedent_support > 0.0 {
                    support / antecedent_support
                } else {
                    0.0
                };
                
                if confidence < min_confidence {
                    continue;
                }
                
                // Calculate DERAR-specific measures
                let stability_score = calculate_stability_score(support, transactions.shape()[0]);
                
                // Mutual information (for simplicity, use average MI of all item pairs)
                let mutual_information = if antecedent.len() == 1 && consequent.len() == 1 {
                    let item_a_count = (0..transactions.shape()[0])
                        .map(|tx| (transactions[[tx, antecedent[0]]] != 0) as usize)
                        .sum();
                    let item_b_count = (0..transactions.shape()[0])
                        .map(|tx| (transactions[[tx, consequent[0]]] != 0) as usize)
                        .sum();
                    calculate_mutual_information(tree, antecedent[0], consequent[0], item_a_count, item_b_count)
                } else {
                    // For multi-item rules, use average MI
                    0.1 // Simplified for now
                };
                
                let tcm = calculate_tcm(&antecedent, &consequent, tree, transactions);
                
                let mut rule = DERARule::new(antecedent, consequent);
                rule.support = support;
                rule.confidence = confidence;
                rule.mutual_information = mutual_information;
                rule.stability_score = stability_score;
                rule.target_concentration_measure = tcm;
                
                rules.push(rule);
            }
        }
    }
    
    rules
}

/// Helper function to calculate pattern support
fn calculate_pattern_support(pattern: &[usize], transactions: ArrayView2<i32>) -> f64 {
    let num_transactions = transactions.shape()[0];
    
    let count = (0..num_transactions)
        .map(|tx_idx| {
            let has_pattern = pattern.iter()
                .all(|&item| transactions[[tx_idx, item]] != 0);
            has_pattern as usize
        })
        .sum::<usize>();
    
    count as f64 / num_transactions as f64
}

/// Main DERAR algorithm implementation
fn derar_algorithm(
    transactions: ArrayView2<i32>, 
    min_support: f64, 
    min_confidence: f64,
    stability_threshold: f64,
    mi_threshold: f64,
    tcm_threshold: f64,
) -> Vec<DERARule> {
    // Step 1: Build Meta-Pattern Tree with statistical tracking
    let mut meta_tree = MetaPatternTree::new();
    
    // First pass: build frequent items and tree
    let frequent_items = generate_frequent_1_itemsets_flat(transactions, min_support);
    let frequent_item_set: std::collections::HashSet<usize> = frequent_items.iter_itemsets()
        .flat_map(|itemset| itemset.iter().cloned())
        .collect();
    
    // Insert transactions into meta-pattern tree
    for tx_idx in 0..transactions.shape()[0] {
        let mut transaction = Vec::new();
        
        for item_idx in 0..transactions.shape()[1] {
            if transactions[[tx_idx, item_idx]] != 0 && frequent_item_set.contains(&item_idx) {
                transaction.push(item_idx);
            }
        }
        
        if !transaction.is_empty() {
            let counts = vec![1; transaction.len()];
            meta_tree.insert_transaction_with_stats(&transaction, &counts);
        }
    }
    
    // Step 2: Mine frequent patterns using our existing FP-Growth
    let frequent_levels = fp_growth_algorithm(transactions, min_support);
    let frequent_patterns: Vec<Vec<usize>> = frequent_levels.iter()
        .flat_map(|level| level.iter_itemsets().map(|itemset| itemset.to_vec()))
        .collect();
    
    // Step 3: Generate rules with DERAR quality measures
    let rules = generate_derar_rules(&frequent_patterns, &meta_tree, transactions, min_confidence);
    
    // Step 4: Apply DERAR's multi-stage filtering
    let filtered_rules = filter_rules_derar(
        rules,
        stability_threshold,
        mi_threshold,
        min_confidence,
        tcm_threshold,
    );
    
    filtered_rules
}

#[pymodule]
fn priors<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    
    /// Memory-optimized Python wrapper for the Apriori algorithm
    /// 
    /// Parameters:
    /// - transactions: 2D binary matrix where rows are transactions and columns are items
    /// - min_support: minimum support threshold (between 0 and 1)
    /// 
    /// Returns:
    /// - List of frequent itemsets (as lists of item indices) for each level
    #[pyfn(m)]
    #[pyo3(name = "apriori")]
    fn apriori_py<'py>(
        py: Python<'py>,
        transactions: PyReadonlyArray2<'py, i32>,
        min_support: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray2<usize>>>> {
        let transactions_view = transactions.as_array();
        let frequent_levels = apriori_algorithm_flat(transactions_view, min_support);
        
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
    
    /// Ultra-fast bit vector Apriori algorithm (for datasets with ≤64 items)
    /// 
    /// Parameters:
    /// - transactions: 2D binary matrix where rows are transactions and columns are items
    /// - min_support: minimum support threshold (between 0 and 1)
    /// 
    /// Returns:
    /// - List of frequent itemsets (as lists of item indices) for each level
    #[pyfn(m)]
    #[pyo3(name = "apriori_bitvector")]
    fn apriori_bitvector_py<'py>(
        py: Python<'py>,
        transactions: PyReadonlyArray2<'py, i32>,
        min_support: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray2<usize>>>> {
        let transactions_view = transactions.as_array();
        
        if transactions_view.shape()[1] > 64 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Bit vector optimization only supports ≤64 items, got {}", transactions_view.shape()[1])
            ));
        }
        
        let frequent_levels = apriori_bitvector(transactions_view, min_support);
        
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
    
    /// DERAR algorithm Python wrapper (Dynamic Extracting of Relevant Association Rules)
    /// 
    /// Parameters:
    /// - transactions: 2D binary matrix where rows are transactions and columns are items
    /// - min_support: minimum support threshold (between 0 and 1)
    /// - min_confidence: minimum confidence threshold (between 0 and 1)
    /// - stability_threshold: minimum stability score threshold (default: 20.0)
    /// - mi_threshold: minimum mutual information threshold (default: 0.1)
    /// - tcm_threshold: minimum target concentration measure threshold (default: 0.5)
    /// 
    /// Returns:
    /// - List of high-quality association rules with statistical measures
    #[pyfn(m)]
    #[pyo3(name = "derar", signature = (transactions, min_support, min_confidence, stability_threshold=None, mi_threshold=None, tcm_threshold=None))]
    fn derar_py<'py>(
        _py: Python<'py>,
        transactions: PyReadonlyArray2<'py, i32>,
        min_support: f64,
        min_confidence: f64,
        stability_threshold: Option<f64>,
        mi_threshold: Option<f64>,
        tcm_threshold: Option<f64>,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, f64, f64, f64, f64, f64)>> {
        let transactions_view = transactions.as_array();
        let rules = derar_algorithm(
            transactions_view, 
            min_support, 
            min_confidence,
            stability_threshold.unwrap_or(20.0),  // Higher stability as per paper
            mi_threshold.unwrap_or(0.1),          // Require meaningful MI
            tcm_threshold.unwrap_or(0.5),         // Higher target concentration
        );
        
        // Convert to Python-friendly format: (antecedent, consequent, support, confidence, MI, stability, TCM)
        let result: Vec<(Vec<usize>, Vec<usize>, f64, f64, f64, f64, f64)> = rules.into_iter()
            .map(|rule| (
                rule.antecedent,
                rule.consequent,
                rule.support,
                rule.confidence,
                rule.mutual_information,
                rule.stability_score,
                rule.target_concentration_measure,
            ))
            .collect();
        
        Ok(result)
    }
    
    /// Memory-optimized support calculation for given itemsets
    /// 
    /// Parameters:
    /// - transactions: 2D binary matrix where rows are transactions and columns are items
    /// - itemsets: 2D array where each row represents an itemset (list of item indices)
    /// 
    /// Returns:
    /// - Array of support values for each itemset
    #[pyfn(m)]
    #[pyo3(name = "calculate_support")]
    fn calculate_support_py<'py>(
        py: Python<'py>,
        transactions: PyReadonlyArray2<'py, i32>,
        itemsets: PyReadonlyArray2<'py, usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let transactions_view = transactions.as_array();
        let itemsets_view = itemsets.as_array();
        
        // Convert to flat storage
        let itemset_size = itemsets_view.shape()[1];
        let mut level = FrequentLevel::with_capacity(itemset_size, itemsets_view.shape()[0]);
        
        for i in 0..itemsets_view.shape()[0] {
            let row = itemsets_view.row(i);
            level.add_itemset(row.to_vec());
        }
        
        let support_values = calculate_support_flat(transactions_view, &level);
        
        Array1::from_vec(support_values).into_pyarray(py)
    }
    
    /// Memory-optimized confidence calculation for association rules
    /// 
    /// Parameters:
    /// - transactions: 2D binary matrix where rows are transactions and columns are items
    /// - antecedent: 1D array representing the antecedent itemset
    /// - consequent: 1D array representing the consequent itemset
    /// 
    /// Returns:
    /// - Confidence value for the rule antecedent -> consequent
    #[pyfn(m)]
    #[pyo3(name = "calculate_confidence")]
    fn calculate_confidence_py(
        transactions: PyReadonlyArray2<i32>,
        antecedent: PyReadonlyArray1<usize>,
        consequent: PyReadonlyArray1<usize>,
    ) -> f64 {
        let transactions_view = transactions.as_array();
        let antecedent_array = antecedent.as_array();
        let consequent_array = consequent.as_array();
        let antecedent_slice = antecedent_array.as_slice().unwrap();
        let consequent_slice = consequent_array.as_slice().unwrap();
        
        calculate_confidence_flat(transactions_view, antecedent_slice, consequent_slice)
    }

    // =============================================================================
    // LAZY FP-GROWTH (Memory-Efficient Streaming Processing)
    // =============================================================================

    // Global storage for lazy FP-Growth processors
    static LAZY_PROCESSORS: Lazy<Mutex<HashMap<usize, LazyFPGrowth>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    static LAZY_NEXT_ID: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

    /// Create a new lazy FP-Growth processor for streaming data
    #[pyfn(m)]
    #[pyo3(name = "create_lazy_fp_growth")]
    fn create_lazy_fp_growth_py() -> PyResult<usize> {
        let mut next_id = LAZY_NEXT_ID.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);
        
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        processors.insert(id, LazyFPGrowth::new());
        Ok(id)
    }

    /// Process a chunk in the counting pass
    #[pyfn(m)]
    #[pyo3(name = "lazy_count_pass")]
    fn lazy_count_pass_py(
        processor_id: usize,
        chunk: PyReadonlyArray2<i32>,
    ) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get_mut(&processor_id) {
            let chunk_view = chunk.as_array();
            processor.count_pass(chunk_view);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    /// Finalize counting pass and get frequent items
    #[pyfn(m)]
    #[pyo3(name = "lazy_finalize_counts")]
    fn lazy_finalize_counts_py<'py>(
        py: Python<'py>,
        processor_id: usize,
        min_support: f64,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get_mut(&processor_id) {
            let frequent_items = processor.finalize_counts(min_support);
            let array = PyArray1::from_vec(py, frequent_items);
            Ok(array)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    /// Process a chunk in the building pass
    #[pyfn(m)]
    #[pyo3(name = "lazy_build_pass")]
    fn lazy_build_pass_py(
        processor_id: usize,
        chunk: PyReadonlyArray2<i32>,
    ) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get_mut(&processor_id) {
            let chunk_view = chunk.as_array();
            processor.build_pass(chunk_view)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    /// Mine patterns from the built FP-tree
    #[pyfn(m)]
    #[pyo3(name = "lazy_mine_patterns")]
    fn lazy_mine_patterns_py<'py>(
        py: Python<'py>,
        processor_id: usize,
        min_support: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray2<usize>>>> {
        let processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get(&processor_id) {
                let frequent_levels = processor.mine_patterns(min_support)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                
                let mut result = Vec::new();
                
                for level in frequent_levels {
                    if level.len() == 0 {
                        continue;
                    }
                    
                    let itemset_size = level.itemset_size;
                    let num_itemsets = level.len();
                    
                    // Create a 2D array: rows = itemsets, cols = items in each itemset
                    let mut data = vec![0usize; num_itemsets * itemset_size];
                    
                    for (itemset_idx, itemset) in level.iter_itemsets().enumerate() {
                        let start_idx = itemset_idx * itemset_size;
                        for (item_idx, &item) in itemset.iter().enumerate() {
                            data[start_idx + item_idx] = item;
                        }
                    }
                    
                    // Reshape flat data into 2D array: rows = itemsets, cols = items per itemset
                    let array = PyArray1::from_vec(py, data).reshape([num_itemsets, itemset_size])?;
                    result.push(array);
                }
                
            Ok(result)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    /// Get statistics about the processor
    #[pyfn(m)]
    #[pyo3(name = "lazy_get_stats")]
    fn lazy_get_stats_py(processor_id: usize) -> PyResult<(usize, usize, usize, usize)> {
        let processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get(&processor_id) {
            let stats = processor.get_stats();
            Ok((
                stats.total_transactions,
                stats.unique_items,
                stats.frequent_items,
                stats.fp_tree_nodes,
            ))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    /// Clean up a processor (free memory)
    #[pyfn(m)]
    #[pyo3(name = "lazy_cleanup")]
    fn lazy_cleanup_py(processor_id: usize) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        processors.remove(&processor_id);
        Ok(())
    }
    
    Ok(())
}