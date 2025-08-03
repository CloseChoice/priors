use numpy::ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

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
    
    Ok(())
}