"""
Tests for utility functions in utils.py.

Essential tests for functions used across the test suite.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import count_itemsets, generate_all_ones_transactions

import priors


def test_count_itemsets_none():
    """Test count_itemsets with None input."""
    assert count_itemsets(None) == 0


def test_count_itemsets_tuple_format():
    """Test count_itemsets with tuple (itemsets_list, supports_list) format."""
    transactions = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)
    result = priors.fp_growth(transactions, 0.3)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert count_itemsets(result) > 0


def test_count_itemsets_list_format():
    """Test count_itemsets with list format."""
    result = [
        np.array([[0], [1], [2]], dtype=np.uint64),
        np.array([[0, 1], [1, 2]], dtype=np.uint64),
    ]
    assert count_itemsets(result) == 5


def test_count_itemsets_single_array():
    """Test count_itemsets with single array."""
    result = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    assert count_itemsets(result) == 3


def test_count_itemsets_invalid():
    """Test count_itemsets with invalid input."""
    assert count_itemsets("invalid") == 0


def test_generate_all_ones_transactions():
    """Test generate_all_ones_transactions creates all-ones matrix."""
    transactions = generate_all_ones_transactions(10, 5)

    assert transactions.shape == (10, 5)
    assert transactions.dtype == np.int32
    assert np.all(transactions == 1)


def test_fp_growth_to_dataframe():
    """Test fp_growth_to_dataframe converts priors result to DataFrame."""
    from utils import fp_growth_to_dataframe

    transactions = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)
    result = priors.fp_growth(transactions, 0.3)

    itemsets_list, supports_list = result
    df = fp_growth_to_dataframe(itemsets_list, supports_list, len(transactions))

    assert isinstance(df, pd.DataFrame)
    assert "support" in df.columns
    assert "itemsets" in df.columns
    assert len(df) > 0
    assert all(isinstance(itemset, frozenset) for itemset in df["itemsets"])
    assert all(0 < s <= 1.0 for s in df["support"])
