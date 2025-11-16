"""
Property-based tests using Hypothesis to verify priors implementation against mlxtend.

These tests generate random binary matrices of varying sizes and verify that our
implementation produces the same results as mlxtend's FP-Growth implementation.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import priors

# Import shared utilities
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import fp_growth_to_dataframe

# Skip all tests if mlxtend is not available
mlxtend = pytest.importorskip("mlxtend")
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth


# ============================================================================
# Hypothesis Strategies
# ============================================================================


def binary_transaction_matrix(
    min_transactions=1,
    max_transactions=100,
    min_items=1,
    max_items=50,
):
    """
    Generate a strategy for binary transaction matrices.

    Returns matrices containing only 0s and 1s, representing transactions.
    """
    return st.integers(min_transactions, max_transactions).flatmap(
        lambda n_trans: st.integers(min_items, max_items).flatmap(
            lambda n_items: arrays(
                dtype=np.int32,
                shape=(n_trans, n_items),
                elements=st.integers(0, 1),
            )
        )
    )


def min_support_strategy():
    """Generate reasonable minimum support values."""
    return st.floats(min_value=0.1, max_value=0.9)


# ============================================================================
# Property-Based Tests
# ============================================================================


@given(
    transactions=binary_transaction_matrix(
        min_transactions=2, max_transactions=50, min_items=2, max_items=20
    ),
    min_support=min_support_strategy(),
)
@settings(max_examples=50, deadline=5000)
def test_fp_growth_matches_mlxtend_random(transactions, min_support):
    """
    Property: priors.fp_growth should produce identical results to mlxtend
    for any binary transaction matrix and any valid minimum support value.
    """
    # Skip if transactions are empty or all zeros
    if transactions.size == 0 or transactions.sum() == 0:
        return

    num_transactions = len(transactions)

    # Run priors
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, num_transactions)

    # Run mlxtend
    df = pd.DataFrame(
        transactions.astype(bool),
        columns=[f"item_{i}" for i in range(transactions.shape[1])],
    )
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)

    # Compare counts
    priors_count = len(priors_result)
    mlxtend_count = len(mlxtend_result)
    assert priors_count == mlxtend_count, (
        f"Itemset count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"
    )

    # Compare itemsets and supports (order-independent)
    priors_set = {
        (frozenset(row["itemsets"]), round(row["support"], 10))
        for _, row in priors_result.iterrows()
    }
    mlxtend_set = {
        (frozenset(row["itemsets"]), round(row["support"], 10))
        for _, row in mlxtend_result.iterrows()
    }

    assert priors_set == mlxtend_set, (
        f"Itemsets mismatch:\n"
        f"Priors only: {priors_set - mlxtend_set}\n"
        f"Mlxtend only: {mlxtend_set - priors_set}"
    )


@given(
    n_trans=st.integers(1, 100),
    n_items=st.integers(1, 30),
    min_support=min_support_strategy(),
)
@settings(max_examples=30, deadline=5000)
def test_fp_growth_all_zeros(n_trans, n_items, min_support):
    """
    Property: Empty transactions (all zeros) should produce no frequent itemsets.
    """
    transactions = np.zeros((n_trans, n_items), dtype=np.int32)

    # Run priors
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, n_trans)

    # Run mlxtend
    df = pd.DataFrame(
        transactions.astype(bool),
        columns=[f"item_{i}" for i in range(transactions.shape[1])],
    )
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)

    # Both should return empty results
    assert len(priors_result) == 0, "All zeros should produce no itemsets (priors)"
    assert len(mlxtend_result) == 0, "All zeros should produce no itemsets (mlxtend)"


@given(
    n_trans=st.integers(1, 100),
    n_items=st.integers(1, 30),
    min_support=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=30, deadline=5000)
def test_fp_growth_all_ones(n_trans, n_items, min_support):
    """
    Property: Transactions with all ones should produce all possible itemsets
    when min_support <= 1.0 (since all items have 100% support).
    """
    transactions = np.ones((n_trans, n_items), dtype=np.int32)

    # Run priors
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, n_trans)

    # Run mlxtend
    df = pd.DataFrame(
        transactions.astype(bool),
        columns=[f"item_{i}" for i in range(transactions.shape[1])],
    )
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)

    # Compare results
    priors_count = len(priors_result)
    mlxtend_count = len(mlxtend_result)
    assert priors_count == mlxtend_count, (
        f"All ones: count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"
    )

    # If min_support <= 1.0, should find all 2^n_items - 1 itemsets
    if min_support <= 1.0 and n_items <= 10:  # Only check for small n_items
        expected_count = 2**n_items - 1
        assert priors_count == expected_count, (
            f"All ones should produce {expected_count} itemsets, got {priors_count}"
        )

    # Verify itemsets match
    priors_set = {
        (frozenset(row["itemsets"]), round(row["support"], 10))
        for _, row in priors_result.iterrows()
    }
    mlxtend_set = {
        (frozenset(row["itemsets"]), round(row["support"], 10))
        for _, row in mlxtend_result.iterrows()
    }

    assert priors_set == mlxtend_set


@given(
    transactions=binary_transaction_matrix(
        min_transactions=5, max_transactions=30, min_items=3, max_items=15
    ),
)
@settings(max_examples=20, deadline=5000)
def test_fp_growth_monotonicity(transactions):
    """
    Property: Lower minimum support should find equal or more itemsets (monotonicity).
    This property should hold for both priors and mlxtend.
    """
    # Skip if transactions are empty or all zeros
    if transactions.size == 0 or transactions.sum() == 0:
        return

    support_levels = [0.6, 0.4, 0.2]
    prev_priors_count = 0
    prev_mlxtend_count = 0

    for min_support in support_levels:
        # Run priors
        itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
        priors_result = fp_growth_to_dataframe(
            itemsets_list, supports_list, len(transactions)
        )
        priors_count = len(priors_result)

        # Run mlxtend
        df = pd.DataFrame(
            transactions.astype(bool),
            columns=[f"item_{i}" for i in range(transactions.shape[1])],
        )
        mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
        mlxtend_count = len(mlxtend_result)

        # Check monotonicity for priors
        assert priors_count >= prev_priors_count, (
            f"Priors: Lower support should find >= itemsets: "
            f"support={min_support}, count={priors_count}, prev={prev_priors_count}"
        )

        # Check monotonicity for mlxtend
        assert mlxtend_count >= prev_mlxtend_count, (
            f"Mlxtend: Lower support should find >= itemsets: "
            f"support={min_support}, count={mlxtend_count}, prev={prev_mlxtend_count}"
        )

        # Verify counts match between implementations
        assert priors_count == mlxtend_count, (
            f"Count mismatch at support={min_support}: "
            f"priors={priors_count}, mlxtend={mlxtend_count}"
        )

        prev_priors_count = priors_count
        prev_mlxtend_count = mlxtend_count


@given(
    base_transactions=binary_transaction_matrix(
        min_transactions=2, max_transactions=20, min_items=2, max_items=10
    ),
    scale_factor=st.integers(1, 10),
    min_support=min_support_strategy(),
)
@settings(max_examples=20, deadline=5000)
def test_fp_growth_scale_invariance(base_transactions, scale_factor, min_support):
    """
    Property: Tiling (repeating) transactions should produce the same itemsets
    at the same support level, since the frequency distribution remains the same.
    """
    # Skip if transactions are empty or all zeros
    if base_transactions.size == 0 or base_transactions.sum() == 0:
        return

    # Scale up by tiling
    scaled_transactions = np.tile(base_transactions, (scale_factor, 1))

    # Run priors on base
    itemsets_list_base, supports_list_base = priors.fp_growth(
        base_transactions, min_support
    )
    priors_result_base = fp_growth_to_dataframe(
        itemsets_list_base, supports_list_base, len(base_transactions)
    )

    # Run priors on scaled
    itemsets_list_scaled, supports_list_scaled = priors.fp_growth(
        scaled_transactions, min_support
    )
    priors_result_scaled = fp_growth_to_dataframe(
        itemsets_list_scaled, supports_list_scaled, len(scaled_transactions)
    )

    # Extract itemsets (ignoring support values which might have tiny floating point diffs)
    base_itemsets = {frozenset(row["itemsets"]) for _, row in priors_result_base.iterrows()}
    scaled_itemsets = {
        frozenset(row["itemsets"]) for _, row in priors_result_scaled.iterrows()
    }

    assert base_itemsets == scaled_itemsets, (
        f"Scaling changed itemsets:\n"
        f"Base only: {base_itemsets - scaled_itemsets}\n"
        f"Scaled only: {scaled_itemsets - base_itemsets}"
    )

    # Also verify with mlxtend
    df_base = pd.DataFrame(
        base_transactions.astype(bool),
        columns=[f"item_{i}" for i in range(base_transactions.shape[1])],
    )
    mlxtend_result_base = mlxtend_fpgrowth(
        df_base, min_support=min_support, use_colnames=False
    )
    mlxtend_base_itemsets = {
        frozenset(row["itemsets"]) for _, row in mlxtend_result_base.iterrows()
    }

    assert base_itemsets == mlxtend_base_itemsets, "Scale invariance: priors != mlxtend"


@given(
    transactions=binary_transaction_matrix(
        min_transactions=3, max_transactions=30, min_items=2, max_items=15
    ),
    min_support=st.floats(min_value=0.15, max_value=0.85),
)
@settings(max_examples=30, deadline=5000)
def test_fp_growth_support_values(transactions, min_support):
    """
    Property: All returned itemsets should have support >= min_support.
    This should be true for both priors and mlxtend.
    """
    # Skip if transactions are empty or all zeros
    if transactions.size == 0 or transactions.sum() == 0:
        return

    num_transactions = len(transactions)

    # Run priors
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, num_transactions)

    # Run mlxtend
    df = pd.DataFrame(
        transactions.astype(bool),
        columns=[f"item_{i}" for i in range(transactions.shape[1])],
    )
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)

    # Check priors support values
    if len(priors_result) > 0:
        min_priors_support = priors_result["support"].min()
        assert min_priors_support >= min_support - 1e-10, (
            f"Priors: Found itemset with support {min_priors_support} < {min_support}"
        )

    # Check mlxtend support values
    if len(mlxtend_result) > 0:
        min_mlxtend_support = mlxtend_result["support"].min()
        assert min_mlxtend_support >= min_support - 1e-10, (
            f"Mlxtend: Found itemset with support {min_mlxtend_support} < {min_support}"
        )

    # Verify all support values match between implementations
    for _, priors_row in priors_result.iterrows():
        itemset = frozenset(priors_row["itemsets"])
        priors_support = priors_row["support"]

        # Find matching itemset in mlxtend results
        mlxtend_row = mlxtend_result[
            mlxtend_result["itemsets"].apply(lambda x: frozenset(x) == itemset)
        ]

        assert len(mlxtend_row) == 1, f"Itemset {itemset} not found in mlxtend results"
        mlxtend_support = mlxtend_row.iloc[0]["support"]

        # Support values should match (within floating point tolerance)
        assert abs(priors_support - mlxtend_support) < 1e-10, (
            f"Support mismatch for {itemset}: "
            f"priors={priors_support}, mlxtend={mlxtend_support}"
        )
