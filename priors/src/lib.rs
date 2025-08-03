use std::collections::{HashMap, HashSet};
use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

/// Represents an itemset as a sorted vector of item indices
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Itemset(Vec<usize>);

impl Itemset {
    fn new(mut items: Vec<usize>) -> Self {
        items.sort();
        Itemset(items)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn contains(&self, item: &usize) -> bool {
        self.0.contains(item)
    }

    fn can_join(&self, other: &Itemset) -> bool {
        if self.len() != other.len() {
            return false;
        }
        
        let k = self.len();
        if k == 0 {
            return false;
        }

        // For k-itemsets, first k-1 items should be the same
        for i in 0..k-1 {
            if self.0[i] != other.0[i] {
                return false;
            }
        }
        
        // Last items should be different
        self.0[k-1] != other.0[k-1]
    }

    fn join(&self, other: &Itemset) -> Option<Itemset> {
        if !self.can_join(other) {
            return None;
        }
        
        let mut joined = self.0.clone();
        joined.push(other.0[other.0.len() - 1]);
        joined.sort();
        Some(Itemset(joined))
    }

    fn subsets_of_size(&self, k: usize) -> Vec<Itemset> {
        if k > self.len() {
            return vec![];
        }
        
        let mut subsets = vec![];
        generate_combinations(&self.0, k, 0, &mut vec![], &mut subsets);
        subsets
    }
}

fn generate_combinations(items: &[usize], k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Itemset>) {
    if current.len() == k {
        result.push(Itemset::new(current.clone()));
        return;
    }
    
    for i in start..items.len() {
        current.push(items[i]);
        generate_combinations(items, k, i + 1, current, result);
        current.pop();
    }
}

/// Calculate support for itemsets given a binary transaction matrix
fn calculate_support(transactions: ArrayView2<i32>, itemsets: &[Itemset]) -> HashMap<Itemset, f64> {
    let num_transactions = transactions.shape()[0] as f64;
    let mut support_map = HashMap::new();
    
    for itemset in itemsets {
        let mut count = 0;
        
        for transaction_idx in 0..transactions.shape()[0] {
            let mut all_present = true;
            for &item in &itemset.0 {
                if transactions[[transaction_idx, item]] == 0 {
                    all_present = false;
                    break;
                }
            }
            if all_present {
                count += 1;
            }
        }
        
        let support = count as f64 / num_transactions;
        support_map.insert(itemset.clone(), support);
    }
    
    support_map
}

/// Generate frequent 1-itemsets
fn generate_frequent_1_itemsets(transactions: ArrayView2<i32>, min_support: f64) -> Vec<Itemset> {
    let num_items = transactions.shape()[1];
    let candidates: Vec<Itemset> = (0..num_items).map(|i| Itemset::new(vec![i])).collect();
    
    let support_map = calculate_support(transactions, &candidates);
    
    candidates.into_iter()
        .filter(|itemset| support_map.get(itemset).unwrap_or(&0.0) >= &min_support)
        .collect()
}

/// Generate candidate itemsets of size k from frequent (k-1)-itemsets
fn generate_candidates(frequent_itemsets: &[Itemset]) -> Vec<Itemset> {
    let mut candidates = vec![];
    
    for i in 0..frequent_itemsets.len() {
        for j in i+1..frequent_itemsets.len() {
            if let Some(candidate) = frequent_itemsets[i].join(&frequent_itemsets[j]) {
                candidates.push(candidate);
            }
        }
    }
    
    candidates
}

/// Prune candidates using the Apriori property
fn prune_candidates(candidates: Vec<Itemset>, frequent_itemsets: &[Itemset]) -> Vec<Itemset> {
    let frequent_set: HashSet<&Itemset> = frequent_itemsets.iter().collect();
    
    candidates.into_iter()
        .filter(|candidate| {
            let subsets = candidate.subsets_of_size(candidate.len() - 1);
            subsets.iter().all(|subset| frequent_set.contains(subset))
        })
        .collect()
}

/// Main Apriori algorithm implementation
fn apriori_algorithm(transactions: ArrayView2<i32>, min_support: f64) -> Vec<Vec<Itemset>> {
    let mut all_frequent_itemsets = vec![];
    
    // Generate frequent 1-itemsets
    let mut frequent_k = generate_frequent_1_itemsets(transactions, min_support);
    
    while !frequent_k.is_empty() {
        all_frequent_itemsets.push(frequent_k.clone());
        
        // Generate candidates
        let candidates = generate_candidates(&frequent_k);
        
        if candidates.is_empty() {
            break;
        }
        
        // Prune candidates
        let pruned_candidates = prune_candidates(candidates, &frequent_k);
        
        if pruned_candidates.is_empty() {
            break;
        }
        
        // Calculate support for candidates
        let support_map = calculate_support(transactions, &pruned_candidates);
        
        // Filter by minimum support
        frequent_k = pruned_candidates.into_iter()
            .filter(|itemset| support_map.get(itemset).unwrap_or(&0.0) >= &min_support)
            .collect();
    }
    
    all_frequent_itemsets
}

/// Calculate confidence for association rules
fn calculate_confidence(transactions: ArrayView2<i32>, antecedent: &Itemset, consequent: &Itemset) -> f64 {
    let combined = {
        let mut items = antecedent.0.clone();
        items.extend(&consequent.0);
        Itemset::new(items)
    };
    
    let antecedent_support = calculate_support(transactions, &[antecedent.clone()])
        .get(antecedent).cloned().unwrap_or(0.0);
    
    let combined_support = calculate_support(transactions, &[combined])
        .values().next().cloned().unwrap_or(0.0);
    
    if antecedent_support == 0.0 {
        0.0
    } else {
        combined_support / antecedent_support
    }
}

#[pymodule]
fn priors<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    
    /// Python wrapper for the Apriori algorithm
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
        let frequent_itemsets = apriori_algorithm(transactions_view, min_support);
        
        let mut result = Vec::new();
        
        for level_itemsets in frequent_itemsets {
            if level_itemsets.is_empty() {
                continue;
            }
            
            let itemset_size = level_itemsets[0].len();
            let num_itemsets = level_itemsets.len();
            
            // Create a 2D array: rows = itemsets, cols = items in each itemset
            let mut data = vec![0usize; num_itemsets * itemset_size];
            
            for (i, itemset) in level_itemsets.iter().enumerate() {
                for (j, &item) in itemset.0.iter().enumerate() {
                    data[i * itemset_size + j] = item;
                }
            }
            
            let array = Array2::from_shape_vec((num_itemsets, itemset_size), data)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to create array"))?;
            
            result.push(array.into_pyarray(py));
        }
        
        Ok(result)
    }
    
    /// Calculate support for given itemsets
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
        
        let itemsets_rust: Vec<Itemset> = (0..itemsets_view.shape()[0])
            .map(|i| {
                let row = itemsets_view.row(i);
                Itemset::new(row.to_vec())
            })
            .collect();
        
        let support_map = calculate_support(transactions_view, &itemsets_rust);
        
        let support_values: Vec<f64> = itemsets_rust.iter()
            .map(|itemset| support_map.get(itemset).cloned().unwrap_or(0.0))
            .collect();
        
        Array1::from_vec(support_values).into_pyarray(py)
    }
    
    /// Calculate confidence for association rules
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
        let antecedent_itemset = Itemset::new(antecedent.as_array().to_vec());
        let consequent_itemset = Itemset::new(consequent.as_array().to_vec());
        
        calculate_confidence(transactions_view, &antecedent_itemset, &consequent_itemset)
    }
    
    Ok(())
}