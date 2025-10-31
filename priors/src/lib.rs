use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{Bound, PyResult, Python, pymodule, types::PyModule};

pub mod fp;
use fp::fp_growth_algorithm;

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

    Ok(())
}
