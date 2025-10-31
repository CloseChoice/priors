use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{Bound, PyResult, Python, pymodule, types::PyModule};
use std::sync::Mutex;
use once_cell::sync::Lazy;
use std::collections::HashMap;

pub mod fp;
use fp::fp_growth_algorithm;
use fp::{StreamingState, count_pass, finalize_counts as fp_finalize_counts,
         build_pass, finalize_building as fp_finalize_building, mine_patterns};

// Global storage for streaming processors
static PROCESSORS: Lazy<Mutex<HashMap<usize, StreamingState>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});
static NEXT_PID: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

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

    // Streaming FP-Growth functions
    #[pyfn(m)]
    #[pyo3(name = "create_lazy_fp_growth")]
    fn create_lazy_fp_growth_py() -> PyResult<usize> {
        let mut pid_lock = NEXT_PID.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
        let pid = *pid_lock;
        *pid_lock += 1;
        drop(pid_lock);

        let mut processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
        processors.insert(pid, StreamingState::new());

        Ok(pid)
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_count_pass")]
    fn lazy_count_pass_py(
        pid: usize,
        transactions: PyReadonlyArray2<i32>,
    ) -> PyResult<()> {
        let transactions_view = transactions.as_array();
        let mut processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        let state = processors.get_mut(&pid)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))?;

        count_pass(state, transactions_view)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_finalize_counts")]
    fn lazy_finalize_counts_py(
        pid: usize,
        min_support: f64,
    ) -> PyResult<()> {
        let mut processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        let state = processors.get_mut(&pid)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))?;

        fp_finalize_counts(state, min_support)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_build_pass")]
    fn lazy_build_pass_py(
        pid: usize,
        transactions: PyReadonlyArray2<i32>,
    ) -> PyResult<()> {
        let transactions_view = transactions.as_array();
        let mut processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        let state = processors.get_mut(&pid)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))?;

        build_pass(state, transactions_view)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_finalize_building")]
    fn lazy_finalize_building_py(pid: usize) -> PyResult<()> {
        let mut processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        let state = processors.get_mut(&pid)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))?;

        fp_finalize_building(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_mine_patterns")]
    fn lazy_mine_patterns_py<'py>(
        py: Python<'py>,
        pid: usize,
        min_support: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray2<usize>>>> {
        let processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        let state = processors.get(&pid)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))?;

        let frequent_levels = mine_patterns(state, min_support)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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
    }

    #[pyfn(m)]
    #[pyo3(name = "lazy_cleanup")]
    fn lazy_cleanup_py(pid: usize) -> PyResult<()> {
        let mut processors = PROCESSORS.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        processors.remove(&pid)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid processor ID"))?;

        Ok(())
    }

    Ok(())
}
