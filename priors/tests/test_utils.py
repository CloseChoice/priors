"""
Tests for utility functions in utils.py.

This test file aims to achieve 100% coverage of the utils module.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import (
    count_itemsets,
    extract_itemsets_from_result,
    generate_all_ones_transactions,
    generate_transactions,
)

import priors


def test_count_itemsets_none():
    """Test count_itemsets with None input."""
    assert count_itemsets(None) == 0


def test_count_itemsets_tuple_format():
    """Test count_itemsets with tuple (itemsets_list, supports_list) format."""
    # Generate some data and run fp_growth
    transactions = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)
    result = priors.fp_growth(transactions, 0.3)

    # Result should be tuple format
    assert isinstance(result, tuple)
    assert len(result) == 2

    count = count_itemsets(result)
    assert count > 0


def test_count_itemsets_list_format():
    """Test count_itemsets with list format."""
    # Create a mock list format result
    result = [
        np.array([[0], [1], [2]], dtype=np.uint64),  # Size-1 itemsets
        np.array([[0, 1], [1, 2]], dtype=np.uint64),  # Size-2 itemsets
    ]

    count = count_itemsets(result)
    assert count == 5  # 3 + 2


def test_count_itemsets_single_array():
    """Test count_itemsets with single array (has shape attribute)."""
    result = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    count = count_itemsets(result)
    assert count == 3


def test_count_itemsets_invalid():
    """Test count_itemsets with invalid input (no shape, not None)."""
    result = "invalid"
    count = count_itemsets(result)
    assert count == 0


@pytest.mark.skip(reason="numpy compatibility issue with reload")
def test_generate_transactions():
    """Test generate_transactions creates correct shape and properties."""
    num_transactions = 100
    num_items = 20
    avg_size = 5

    transactions = generate_transactions(num_transactions, num_items, avg_size, seed=42)

    assert transactions.shape == (num_transactions, num_items)
    assert transactions.dtype == np.int32
    assert np.all((transactions == 0) | (transactions == 1))  # Binary values only

    # Check that we have some transactions
    assert transactions.sum() > 0


def test_generate_all_ones_transactions():
    """Test generate_all_ones_transactions creates all-ones matrix."""
    num_transactions = 10
    num_items = 5

    transactions = generate_all_ones_transactions(num_transactions, num_items)

    assert transactions.shape == (num_transactions, num_items)
    assert transactions.dtype == np.int32
    assert np.all(transactions == 1)


def test_extract_itemsets_from_result_none():
    """Test extract_itemsets_from_result with None."""
    from utils import extract_itemsets_from_result as extract_func

    result = extract_func(None)
    assert result == set()


def test_extract_itemsets_from_result_list():
    """Test extract_itemsets_from_result with list format."""
    from utils import extract_itemsets_from_result as extract_func

    # Create mock result
    result = [
        np.array([[0], [1], [2]], dtype=np.uint64),  # Size-1 itemsets
        np.array([[0, 1], [1, 2]], dtype=np.uint64),  # Size-2 itemsets
    ]

    itemsets = extract_func(result)
    assert len(itemsets) == 5
    assert (0,) in itemsets
    assert (1,) in itemsets
    assert (2,) in itemsets
    assert (0, 1) in itemsets
    assert (1, 2) in itemsets


def test_extract_itemsets_from_mlxtend():
    """Test extract_itemsets_from_mlxtend with mlxtend DataFrame format."""
    pytest.importorskip("mlxtend")
    from utils import extract_itemsets_from_mlxtend

    # Create mock mlxtend result
    mlxtend_result = pd.DataFrame(
        {
            "support": [0.6, 0.5, 0.4],
            "itemsets": [frozenset([0]), frozenset([1]), frozenset([0, 1])],
        }
    )

    itemsets = extract_itemsets_from_mlxtend(mlxtend_result)
    assert len(itemsets) == 3
    assert (0,) in itemsets
    assert (1,) in itemsets
    assert (0, 1) in itemsets


def test_extract_itemsets_from_mlxtend_empty():
    """Test extract_itemsets_from_mlxtend with empty DataFrame."""
    from utils import extract_itemsets_from_mlxtend

    mlxtend_result = pd.DataFrame({"support": [], "itemsets": []})
    itemsets = extract_itemsets_from_mlxtend(mlxtend_result)
    assert itemsets == set()


def test_extract_itemsets_from_mlxtend_none():
    """Test extract_itemsets_from_mlxtend with None."""
    from utils import extract_itemsets_from_mlxtend

    itemsets = extract_itemsets_from_mlxtend(None)
    assert itemsets == set()


def test_extract_itemsets_from_efficient_apriori():
    """Test extract_itemsets_from_efficient_apriori with efficient_apriori format."""
    from utils import extract_itemsets_from_efficient_apriori

    # Create mock efficient_apriori result
    ea_result = {
        1: [frozenset([0]), frozenset([1]), frozenset([2])],
        2: [frozenset([0, 1]), frozenset([1, 2])],
    }

    itemsets = extract_itemsets_from_efficient_apriori(ea_result)
    assert len(itemsets) == 5
    assert (0,) in itemsets
    assert (1,) in itemsets
    assert (2,) in itemsets
    assert (0, 1) in itemsets
    assert (1, 2) in itemsets


def test_extract_itemsets_from_efficient_apriori_empty():
    """Test extract_itemsets_from_efficient_apriori with empty dict."""
    from utils import extract_itemsets_from_efficient_apriori

    itemsets = extract_itemsets_from_efficient_apriori({})
    assert itemsets == set()


def test_extract_itemsets_from_efficient_apriori_none():
    """Test extract_itemsets_from_efficient_apriori with None/False."""
    from utils import extract_itemsets_from_efficient_apriori

    itemsets = extract_itemsets_from_efficient_apriori(None)
    assert itemsets == set()


def test_extract_itemsets_from_priors_list():
    """Test extract_itemsets_from_priors with list format."""
    from utils import extract_itemsets_from_priors

    # Create mock priors result (old list format)
    result = [
        np.array([[0], [1], [2]], dtype=np.uint64),  # Size-1 itemsets
        np.array([[0, 1], [1, 2]], dtype=np.uint64),  # Size-2 itemsets
    ]

    itemsets = extract_itemsets_from_priors(result)
    # This function returns (level_idx, i) tuples, not actual itemsets
    assert len(itemsets) == 5
    assert (0, 0) in itemsets  # First itemset at level 0
    assert (0, 1) in itemsets  # Second itemset at level 0
    assert (0, 2) in itemsets  # Third itemset at level 0
    assert (1, 0) in itemsets  # First itemset at level 1
    assert (1, 1) in itemsets  # Second itemset at level 1


def test_extract_itemsets_from_priors_none():
    """Test extract_itemsets_from_priors with None."""
    from utils import extract_itemsets_from_priors

    itemsets = extract_itemsets_from_priors(None)
    assert itemsets == set()


@pytest.mark.skip(reason="numpy/pandas compatibility issue with reload")
def test_fp_growth_to_dataframe():
    """Test fp_growth_to_dataframe converts priors result to DataFrame."""
    from utils import fp_growth_to_dataframe

    # Create sample data
    transactions = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)
    result = priors.fp_growth(transactions, 0.3)

    itemsets_list, supports_list = result
    df = fp_growth_to_dataframe(itemsets_list, supports_list, len(transactions))

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert "support" in df.columns
    assert "itemsets" in df.columns
    assert len(df) > 0

    # Check that itemsets are frozensets
    assert all(isinstance(itemset, frozenset) for itemset in df["itemsets"])

    # Check that support values are in valid range
    assert all(0 < s <= 1.0 for s in df["support"])


@pytest.mark.skip(reason="numpy/pandas compatibility issue with reload")
def test_fp_growth_to_dataframe_sorting():
    """Test that fp_growth_to_dataframe sorts results correctly."""
    from utils import fp_growth_to_dataframe

    # Create simple mock data
    itemsets_list = [
        np.array([[0]], dtype=np.uint64),
    ]
    supports_list = [[50]]  # Counts
    num_transactions = 100

    df = fp_growth_to_dataframe(itemsets_list, supports_list, num_transactions)

    # Check basic structure
    assert len(df) == 1
    assert df.iloc[0]["support"] == 0.5
    assert df.iloc[0]["itemsets"] == frozenset([0])


def test_extract_itemsets_from_result_edge_case():
    """Test extract_itemsets_from_result with level without __getitem__."""
    from utils import extract_itemsets_from_result as extract_func

    # Create a mock object that has shape but no __getitem__
    class MockLevel:
        def __init__(self):
            self.shape = (2,)

    result = [MockLevel()]
    itemsets = extract_func(result)

    # Should use range(level_idx + 1) fallback
    assert (0,) in itemsets
