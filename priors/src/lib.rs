use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use once_cell::sync::Lazy;
use pyo3::{Bound, PyResult, Python, pymodule, types::PyModule};
use std::collections::HashMap;
use std::sync::Mutex;

pub mod fp;
use fp::{LazyFPGrowth, fp_growth_algorithm};

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

#[pymodule]
fn priors<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
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
    fn lazy_count_pass_py(processor_id: usize, chunk: PyReadonlyArray2<i32>) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get_mut(&processor_id) {
            let chunk_view = chunk.as_array();
            processor.count_pass(chunk_view);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid processor ID",
            ))
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
            Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid processor ID",
            ))
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_build_pass")]
    fn lazy_build_pass_py(processor_id: usize, chunk: PyReadonlyArray2<i32>) -> PyResult<()> {
        let mut processors = LAZY_PROCESSORS.lock().unwrap();
        if let Some(processor) = processors.get_mut(&processor_id) {
            let chunk_view = chunk.as_array();
            processor.build_pass(chunk_view);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid processor ID",
            ))
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
            let frequent_levels = processor
                .mine_patterns(min_support)
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

                let array =
                    Array2::from_shape_vec((num_itemsets, itemset_size), data).map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err("Failed to create array")
                    })?;

                result.push(array.into_pyarray(py));
            }
            Ok(result)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid processor ID",
            ))
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
            Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid processor ID",
            ))
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
