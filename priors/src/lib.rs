use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

pub mod fp;
use fp::{fp_growth_algorithm, LazyFPGrowth};

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

#[pymodule]
fn priors<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    /// FP-Growth algorithm Python wrapper
    ///
    /// Finds all frequent itemsets in a transactional database using the FP-Growth algorithm.
    ///
    /// Parameters
    /// ----------
    /// transactions : numpy.ndarray (2D, int32)
    ///     Binary transaction matrix where:
    ///     - Rows represent transactions
    ///     - Columns represent items
    ///     - Values are 0 (item not present) or 1 (item present)
    ///
    /// Example:
    ///
    /// ```text
    /// [[1, 0, 1, 0],  # Transaction 0: items 0 and 2
    ///  [1, 1, 0, 0],  # Transaction 1: items 0 and 1
    ///  [0, 1, 1, 1]]  # Transaction 2: items 1, 2, and 3
    /// ```
    ///
    /// min_support : float
    ///     Minimum support threshold (between 0.0 and 1.0).
    ///     An itemset is considered frequent if it appears in at least
    ///     (min_support * num_transactions) transactions.
    ///
    /// Example: min_support=0.5 means an itemset must appear in at least
    /// 50% of all transactions.
    ///
    /// Returns
    /// -------
    /// list of numpy.ndarray
    ///     List of frequent itemsets grouped by size:
    ///     - result[0]: All frequent 1-itemsets (2D array: rows=itemsets, cols=items)
    ///     - result[1]: All frequent 2-itemsets
    ///     - result[2]: All frequent 3-itemsets
    ///     - etc.
    ///
    /// Each array has shape (num_itemsets, itemset_size).
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> import priors
    /// >>>
    /// >>> # Create transaction data
    /// >>> transactions = np.array([
    /// ...     [1, 1, 0, 0, 1],  # Transaction with items A, B, E
    /// ...     [1, 1, 1, 0, 0],  # Transaction with items A, B, C
    /// ...     [1, 0, 1, 1, 0],  # Transaction with items A, C, D
    /// ...     [0, 1, 1, 0, 0],  # Transaction with items B, C
    /// ... ], dtype=np.int32)
    /// >>>
    /// >>> # Find frequent itemsets with 50% minimum support
    /// >>> result = priors.fp_growth(transactions, min_support=0.5)
    /// >>>
    /// >>> # result[0] contains all frequent 1-itemsets
    /// >>> print("Frequent 1-itemsets:", result[0])
    /// >>> # result[1] contains all frequent 2-itemsets
    /// >>> print("Frequent 2-itemsets:", result[1])
    ///
    /// Notes
    /// -----
    /// The FP-Growth algorithm is more efficient than Apriori for dense datasets
    /// as it:
    /// - Uses a compact FP-Tree data structure (prefix tree compression)
    /// - Avoids explicit candidate generation
    /// - Employs a divide-and-conquer strategy
    /// - Parallelizes mining of different items (via Rayon)
    ///
    /// Time Complexity: O(|DB| + |F|) where |DB| is database size and |F| is
    ///                  number of frequent itemsets
    /// Space Complexity: O(|T|) where |T| is the size of the FP-Tree
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

    static LAZY_PROCESSORS: Lazy<Mutex<HashMap<usize, LazyFPGrowth>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    static LAZY_NEXT_ID: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

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
            Ok(frequent_items.into_pyarray(py))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_build_pass")]
    fn lazy_build_pass_py(
        processor_id: usize,
        chunk: PyReadonlyArray2<i32>,
    ) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get_mut(&processor_id) {
            let chunk_view = chunk.as_array();
            processor.build_pass(chunk_view);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

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
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let mut result = Vec::new();
            for level in frequent_levels {
                if level.len() == 0 {
                    continue;
                }

                let itemset_size = level.itemset_size;
                let num_itemsets = level.len();
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
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

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
                stats.tree_nodes,
            ))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_cleanup")]
    fn lazy_cleanup_py(processor_id: usize) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        processors.remove(&processor_id);
        Ok(())
    }

    Ok(())
}
