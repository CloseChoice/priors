"""
Comprehensive comparison tests between priors FP-Growth and other libraries.
Tests correctness and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
import priors
from typing import Dict, List, Tuple, Optional, Set

# Import shared utilities
from conftest import (
    count_itemsets,
    generate_transactions,
)


# ============================================================================
# Correctness Tests
# ============================================================================

# Test correctness by comparing with established libraries.

def test_fpgrowth_vs_mlxtend_basic():
    """Compare priors FP-Growth with mlxtend on basic dataset."""
    import pandas.testing as tm
    mlxtend = pytest.importorskip("mlxtend")
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

    # Import the conversion utility
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import fp_growth_to_dataframe

    # Create simple test data
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)

    min_support = 0.4

    # Run priors - now returns (itemsets_list, supports_list) tuple
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, len(transactions))

    # Run mlxtend
    df = pd.DataFrame(transactions.astype(bool),
                     columns=[f'item_{i}' for i in range(transactions.shape[1])])
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)

    # Debug output
    print("\n=== PRIORS RESULT ===")
    print(priors_result)
    print("\n=== MLXTEND RESULT ===")
    print(mlxtend_result)

    # Compare DataFrames
    priors_count = len(priors_result)
    mlxtend_count = len(mlxtend_result)
    assert priors_count == mlxtend_count, \
        f"Itemset count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"

    # Compare itemsets and supports (order-independent)
    priors_set = set((frozenset(row['itemsets']), row['support'])
                     for _, row in priors_result.iterrows())
    mlxtend_set = set((frozenset(row['itemsets']), row['support'])
                      for _, row in mlxtend_result.iterrows())

    assert priors_set == mlxtend_set, \
        f"Itemsets mismatch:\nPriors only: {priors_set - mlxtend_set}\nMlxtend only: {mlxtend_set - priors_set}"

def test_fpgrowth_vs_efficient_apriori_basic():
    """Compare priors FP-Growth with efficient_apriori."""
    try:
        import efficient_apriori
    except ImportError:
        pytest.skip("efficient_apriori not available")

    # Import the conversion utility
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import fp_growth_to_dataframe

    # Create test data
    transactions = generate_transactions(50, 10, 4, seed=123)
    min_support = 0.2
    num_transactions = len(transactions)

    # Run priors
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, num_transactions)

    # Run efficient_apriori
    transactions_list = [tuple(np.where(row)[0]) for row in transactions]
    ea_itemsets, ea_rules = efficient_apriori.apriori(transactions_list, min_support=min_support)

    # Convert efficient_apriori results to dictionary: {itemset: support}
    # ea_itemsets format: {1: {(item,): count}, 2: {(item1, item2): count}, ...}
    ea_dict = {}
    if ea_itemsets:
        for size, itemsets_dict in ea_itemsets.items():
            for itemset, count in itemsets_dict.items():
                support = count / num_transactions
                ea_dict[frozenset(itemset)] = support

    # Convert priors results to dictionary: {itemset: support}
    priors_dict = {frozenset(row['itemsets']): row['support']
                   for _, row in priors_result.iterrows()}

    if ea_itemsets:
        for size in sorted(ea_itemsets.keys()):
            print(f"Size {size}:")
            for itemset, count in sorted(ea_itemsets[size].items()):
                support = count / num_transactions
                print(f"  {itemset}: support={support:.3f} (count={count})")

    # Compare counts
    priors_count = len(priors_result)
    ea_count = sum(len(itemsets) for itemsets in ea_itemsets.values()) if ea_itemsets else 0
    assert priors_count == ea_count, \
        f"Itemset count mismatch: priors={priors_count}, efficient_apriori={ea_count}"

    # Compare dictionaries: itemsets and their support values
    assert priors_dict == ea_dict, \
        f"Itemsets mismatch:\nPriors only: {set(priors_dict.keys()) - set(ea_dict.keys())}\n" \
        f"Efficient_apriori only: {set(ea_dict.keys()) - set(priors_dict.keys())}\n" \
        f"Different supports: {[(k, priors_dict[k], ea_dict[k]) for k in priors_dict.keys() & ea_dict.keys() if priors_dict[k] != ea_dict[k]]}"

def test_fpgrowth_vs_mlxtend_medium():
    """Compare with mlxtend on medium-sized dataset."""
    import pandas.testing as tm
    mlxtend = pytest.importorskip("mlxtend")
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

    # Import the conversion utility
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import fp_growth_to_dataframe

    # Generate medium dataset
    transactions = generate_transactions(200, 20, 6, seed=456)
    min_support = 0.1

    # Run priors
    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    priors_result = fp_growth_to_dataframe(itemsets_list, supports_list, len(transactions))

    # Run mlxtend
    df = pd.DataFrame(transactions.astype(bool),
                     columns=[f'item_{i}' for i in range(transactions.shape[1])])
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)

    # Compare results
    priors_count = len(priors_result)
    mlxtend_count = len(mlxtend_result)
    assert priors_count == mlxtend_count, \
        f"Itemset count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"

    # Compare itemsets and supports (order-independent)
    priors_set = set((frozenset(row['itemsets']), row['support'])
                     for _, row in priors_result.iterrows())
    mlxtend_set = set((frozenset(row['itemsets']), row['support'])
                      for _, row in mlxtend_result.iterrows())

    assert priors_set == mlxtend_set, \
        f"Itemsets mismatch:\nPriors only: {priors_set - mlxtend_set}\nMlxtend only: {mlxtend_set - priors_set}"


# ============================================================================
# Scale Tests
# ============================================================================

# Test correctness at different scales using same pattern.

def test_fpgrowth_consistent_across_scales():
    """Test that same pattern gives same itemsets regardless of scale."""
    # Use a simple, deterministic pattern
    base_pattern = np.array([
        [1, 1, 0, 0],  # Items 0,1
        [1, 1, 0, 0],  # Items 0,1
        [0, 0, 1, 1],  # Items 2,3
        [0, 0, 1, 1],  # Items 2,3
    ], dtype=np.int32)

    min_support = 0.5  # 50% - items 0,1 together (50%) and 2,3 together (50%)

    # Test at different scales
    scales = [1, 10, 100, 1000]
    baseline_count = None

    for scale in scales:
        transactions = np.tile(base_pattern, (scale, 1))
        itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets((itemsets_list, supports_list))

        if baseline_count is None:
            baseline_count = itemset_count
            assert itemset_count > 0, f"Should find itemsets with this pattern"
        else:
            assert itemset_count == baseline_count, \
                f"Scale {scale}: found {itemset_count} itemsets, expected {baseline_count}"

def test_fpgrowth_different_supports():
    """Test that lower support finds monotonically more itemsets."""
    transactions = generate_transactions(500, 30, 8, seed=131415)

    support_levels = [0.1, 0.05, 0.02, 0.01]
    prev_count = 0

    for support in support_levels:
        itemsets_list, supports_list = priors.fp_growth(transactions, support)
        itemset_count = count_itemsets((itemsets_list, supports_list))

        # Lower support should find more or equal itemsets (monotonic property)
        assert itemset_count >= prev_count, \
            f"Support {support}: {itemset_count} < {prev_count} from higher support"

        prev_count = itemset_count

def test_fpgrowth_empty_transactions():
    """Test edge case with empty transaction list."""
    transactions = np.array([], dtype=np.int32).reshape(0, 10)
    min_support = 0.1

    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    itemset_count = count_itemsets((itemsets_list, supports_list))

    assert itemset_count == 0, f"Empty transactions should return 0 itemsets, got {itemset_count}"


# ============================================================================
# Deterministic Tests
# ============================================================================

# Test with known datasets and expected results.

def test_simple_known_result():
    """Test with a simple dataset where we know exact expected itemsets."""
    # 3 identical transactions, each containing items 0 and 1
    # At 100% support, we expect:
    # - 1-itemsets: {0}, {1}
    # - 2-itemsets: {0,1}
    # Total: 3 frequent itemsets
    transactions = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ], dtype=np.int32)
    min_support = 1.0  # 100%

    itemsets_list, supports_list = priors.fp_growth(transactions, min_support)
    itemset_count = count_itemsets((itemsets_list, supports_list))

    # Expected: {0}, {1}, {0,1} = 3 itemsets total
    assert itemset_count == 3, f"Expected 3 itemsets, got {itemset_count}"

def test_known_result_with_scaling():
    """Test that scaling identical transactions produces consistent results."""
    # Pattern where all items have same frequency
    base = np.array([
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=np.int32)

    min_support = 1.0  # 100%

    # Run on base - all items appear 100%, so we expect 2^3-1=7 itemsets
    itemsets_list_base, supports_list_base = priors.fp_growth(base, min_support)
    count_base = count_itemsets((itemsets_list_base, supports_list_base))

    # Run on scaled version (100x)
    scaled = np.tile(base, (100, 1))
    itemsets_list_scaled, supports_list_scaled = priors.fp_growth(scaled, min_support)
    count_scaled = count_itemsets((itemsets_list_scaled, supports_list_scaled))

    # Should find same patterns - all items still at 100%
    assert count_base == count_scaled, \
        f"Scaling changed result: base={count_base}, scaled={count_scaled}"

def test_support_threshold_filtering():
    """Test that support threshold correctly filters itemsets."""
    # 10 transactions where item 0 appears 5 times (50%)
    transactions = np.array([
        [1, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 0, 0],  # 2
        [1, 0, 0],  # 3
        [1, 0, 0],  # 4
        [0, 1, 1],  # 5
        [0, 1, 1],  # 6
        [0, 1, 1],  # 7
        [0, 1, 1],  # 8
        [0, 1, 1],  # 9
    ], dtype=np.int32)

    # At 60% support, only items 1,2 should be frequent (50% for item 0)
    itemsets_list_60, supports_list_60 = priors.fp_growth(transactions, 0.6)
    count_60 = count_itemsets((itemsets_list_60, supports_list_60))

    # At 50% support, items 0,1,2 should all be frequent
    itemsets_list_50, supports_list_50 = priors.fp_growth(transactions, 0.5)
    count_50 = count_itemsets((itemsets_list_50, supports_list_50))

    # Lower support should find more itemsets
    assert count_50 > count_60, \
        f"50% support should find more itemsets than 60%: {count_50} vs {count_60}"

def test_scalable_known_results():
    """Test with scalable pattern where we know exact results for any size.

    Pattern: Repeating block of 10 transactions where:
    - Item A appears in transactions 0-4 (50%)
    - Item B appears in transactions 0-6 (70%)
    - Item C appears in transactions 0-9 (100%)

    This pattern is scale-invariant: 10, 100, 1000, 10000 transactions
    all have the same frequency distribution.
    """
    def create_pattern(num_blocks):
        """Create num_blocks * 10 transactions with known pattern."""
        base_block = np.array([
            [1, 1, 1],  # 0: A, B, C
            [1, 1, 1],  # 1: A, B, C
            [1, 1, 1],  # 2: A, B, C
            [1, 1, 1],  # 3: A, B, C
            [1, 1, 1],  # 4: A, B, C
            [0, 1, 1],  # 5: B, C (no A)
            [0, 1, 1],  # 6: B, C (no A)
            [0, 0, 1],  # 7: C (no A, B)
            [0, 0, 1],  # 8: C (no A, B)
            [0, 0, 1],  # 9: C (no A, B)
        ], dtype=np.int32)
        return np.tile(base_block, (num_blocks, 1))

    # Test at different scales
    for num_blocks in [1, 10, 100, 1000]:
        transactions = create_pattern(num_blocks)
        num_trans = num_blocks * 10

        # At 100% support: only item C (100%)
        # Expected: {C} = 1 itemset
        itemsets_list_100, supports_list_100 = priors.fp_growth(transactions, 1.0)
        count_100 = count_itemsets((itemsets_list_100, supports_list_100))
        assert count_100 == 1, \
            f"Scale {num_trans}: 100% support should find exactly 1 itemset, got {count_100}"

        # At 70% support: items B (70%) and C (100%)
        # Expected: {B}, {C}, {B,C} = 3 itemsets
        itemsets_list_70, supports_list_70 = priors.fp_growth(transactions, 0.7)
        count_70 = count_itemsets((itemsets_list_70, supports_list_70))
        assert count_70 == 3, \
            f"Scale {num_trans}: 70% support should find exactly 3 itemsets, got {count_70}"

        # At 50% support: items A (50%), B (70%), C (100%)
        # Expected: {A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C} = 7 itemsets (2^3-1)
        itemsets_list_50, supports_list_50 = priors.fp_growth(transactions, 0.5)
        count_50 = count_itemsets((itemsets_list_50, supports_list_50))
        assert count_50 == 7, \
            f"Scale {num_trans}: 50% support should find exactly 7 itemsets, got {count_50}"
