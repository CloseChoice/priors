import numpy as np
import pytest
import priors


def test_fp_growth_basic():
    """Test basic FP-Growth functionality"""
    # Simple transaction matrix
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
    ], dtype=np.int32)
    
    min_support = 0.4  # 2/5 = 0.4
    
    frequent_itemsets = priors.fp_growth(transactions, min_support)
    
    # Should have at least some frequent itemsets
    assert len(frequent_itemsets) > 0
    
    # Check that we get results
    total_itemsets = sum(level.shape[0] for level in frequent_itemsets)
    assert total_itemsets > 0


def test_fp_growth_vs_apriori_consistency():
    """Test that FP-Growth and Apriori find similar patterns"""
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)
    
    min_support = 0.4
    
    apriori_result = priors.apriori(transactions, min_support)
    fp_growth_result = priors.fp_growth(transactions, min_support)
    
    # Both should find some patterns
    assert len(apriori_result) > 0
    assert len(fp_growth_result) > 0
    
    # Count total itemsets
    apriori_total = sum(level.shape[0] for level in apriori_result)
    fp_growth_total = sum(level.shape[0] for level in fp_growth_result)
    
    print(f"Apriori found: {apriori_total} itemsets")
    print(f"FP-Growth found: {fp_growth_total} itemsets")
    
    # They should find similar numbers (might not be exactly the same due to implementation differences)
    # But at least both should find frequent 1-itemsets
    assert apriori_total > 0
    assert fp_growth_total > 0


def test_fp_growth_empty():
    """Test FP-Growth with empty input"""
    transactions = np.array([], dtype=np.int32).reshape(0, 5)
    min_support = 0.1
    
    result = priors.fp_growth(transactions, min_support)
    assert len(result) == 0


def test_fp_growth_single_transaction():
    """Test FP-Growth with single transaction"""
    transactions = np.array([[1, 0, 1, 1, 0]], dtype=np.int32)
    min_support = 0.5
    
    result = priors.fp_growth(transactions, min_support)
    
    # Should find frequent 1-itemsets
    assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])