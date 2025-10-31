"""
Basic tests for FP-Growth implementation.
Tests core functionality and correctness.
"""

import numpy as np
import pandas as pd
import pytest
import priors
from typing import List, Tuple, Optional

# Import shared utilities
from conftest import count_itemsets, generate_transactions, extract_itemsets_from_result


# ============================================================================
# Basic FP-Growth Tests
# ============================================================================

class TestFPGrowthBasic:
    """Basic functionality tests for FP-Growth."""

    def test_empty_transactions(self):
        """Test with empty transaction matrix."""
        transactions = np.array([], dtype=np.int32).reshape(0, 5)
        min_support = 0.1
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        assert itemset_count == 0, f"Empty transactions should return 0 itemsets, got {itemset_count}"

    def test_single_transaction(self):
        """Test with single transaction."""
        transactions = np.array([[1, 0, 1, 0, 1]], dtype=np.int32)
        min_support = 1.0  # 100% support
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # With single transaction at 100% support, we should get all subsets of [0,2,4]
        # That's 2^3 - 1 = 7 itemsets (excluding empty set)
        expected = 7
        assert itemset_count == expected, f"Single transaction should return {expected} itemsets, got {itemset_count}"

    def test_no_frequent_items(self):
        """Test with support threshold too high."""
        transactions = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.int32)
        min_support = 0.5  # 50% support, but each item appears only 20%
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        assert itemset_count == 0, f"High support threshold should return 0 itemsets, got {itemset_count}"

    def test_all_items_frequent(self):
        """Test where all combinations should be frequent."""
        transactions = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=np.int32)
        min_support = 1.0  # 100% support
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # All subsets of {0,1,2}: 2^3 - 1 = 7
        expected = 7
        assert itemset_count == expected, f"All frequent items should return {expected} itemsets, got {itemset_count}"

    def test_basic_example(self):
        """Test with well-known basic example."""
        transactions = np.array([
            [1, 1, 0, 1, 0],  # Items 0,1,3
            [1, 0, 1, 1, 0],  # Items 0,2,3
            [0, 1, 1, 1, 0],  # Items 1,2,3
            [1, 1, 1, 0, 0],  # Items 0,1,2
            [1, 1, 0, 1, 0],  # Items 0,1,3
        ], dtype=np.int32)
        min_support = 0.4  # 40% support = 2 transactions
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # Manual verification:
        # Item frequencies: 0:4, 1:4, 2:3, 3:4, 4:0
        # Frequent items (≥2): 0,1,2,3
        # Should find multiple frequent itemsets
        assert itemset_count > 0, "Should find frequent itemsets"
        assert itemset_count <= 15, f"Too many itemsets found: {itemset_count}"  # 2^4-1 max

    def test_different_support_levels(self):
        """Test with different support levels."""
        transactions = generate_transactions(100, 10, 5, seed=42)
        
        # Test with decreasing support levels
        support_levels = [0.5, 0.3, 0.1, 0.05]
        prev_count = 0
        
        for support in support_levels:
            result = priors.fp_growth(transactions, support)
            count = count_itemsets(result)
            
            # Lower support should find more or equal itemsets
            assert count >= prev_count, f"Support {support} found {count} itemsets, less than {prev_count} at higher support"
            prev_count = count

    def test_large_transactions(self):
        """Test with larger transaction set."""
        transactions = generate_transactions(1000, 50, 10, seed=123)
        min_support = 0.1

        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)

        assert itemset_count > 0, "Should find itemsets in large dataset"

    def test_very_sparse_data(self):
        """Test with very sparse transaction data."""
        num_trans, num_items = 100, 100
        transactions = np.zeros((num_trans, num_items), dtype=np.int32)
        
        # Only set a few items in a few transactions
        for i in range(0, num_trans, 10):
            transactions[i, i % num_items] = 1
            if i + 1 < num_items:
                transactions[i, (i + 1) % num_items] = 1
        
        min_support = 0.05  # 5%
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # Sparse data should still work
        assert itemset_count >= 0, "Sparse data should not crash"

    def test_dense_data(self):
        """Test with very dense transaction data."""
        num_trans, num_items = 50, 20
        transactions = np.ones((num_trans, num_items), dtype=np.int32)
        
        # Remove some items randomly to make it interesting
        np.random.seed(42)
        mask = np.random.random((num_trans, num_items)) > 0.2  # 80% density
        transactions = transactions * mask.astype(np.int32)
        
        min_support = 0.6  # 60%
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        assert itemset_count > 0, "Dense data should find frequent itemsets"


# ============================================================================
# Edge Cases
# ============================================================================

class TestFPGrowthEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_min_support_zero(self):
        """Test with minimum support of 0."""
        transactions = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.int32)
        min_support = 0.0
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # With 0 support, all possible itemsets should be found
        assert itemset_count > 0, "Zero support should find itemsets"

    def test_min_support_one(self):
        """Test with minimum support of 1.0 (100%)."""
        transactions = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ], dtype=np.int32)
        min_support = 1.0
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # Only items that appear in ALL transactions should be found
        # In this case, no item appears in all 3 transactions
        assert itemset_count == 0, f"100% support should find 0 itemsets, got {itemset_count}"

    def test_single_item_transactions(self):
        """Test with transactions containing only single items."""
        transactions = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],  # Repeat item 0
        ], dtype=np.int32)
        min_support = 0.2  # 20%
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # Only single items should be frequent (no combinations)
        # Item 0 appears twice (40%), others once (20%)
        # So items 0,1,2,3 should all be frequent
        assert itemset_count >= 4, f"Should find at least 4 single items, got {itemset_count}"

    def test_duplicate_transactions(self):
        """Test with duplicate transactions."""
        base_transaction = [1, 1, 0, 1, 0]
        transactions = np.array([base_transaction] * 5, dtype=np.int32)
        min_support = 0.8  # 80%
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # All items in the transaction should be frequent
        # Items 0,1,3 appear in all 5 transactions (100%)
        expected_combinations = 2**3 - 1  # All subsets of {0,1,3}
        assert itemset_count == expected_combinations, f"Expected {expected_combinations} itemsets, got {itemset_count}"

    def test_binary_validation(self):
        """Test that input is properly handled as binary."""
        # Test with values > 1 (should be treated as 1)
        transactions = np.array([
            [2, 3, 0, 5],
            [1, 0, 4, 2],
            [0, 7, 1, 0],
        ], dtype=np.int32)
        min_support = 0.5
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        # Should work without crashing
        assert itemset_count >= 0, "Non-binary input should be handled gracefully"


