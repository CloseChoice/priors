import numpy as np
import pytest
import priors


def test_bitvector_basic():
    """Test basic bit vector functionality"""
    # Simple transaction matrix with ≤64 items
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
    ], dtype=np.int32)
    
    min_support = 0.4  # 2/5 = 0.4
    
    frequent_itemsets = priors.apriori_bitvector(transactions, min_support)
    
    # Should have at least some frequent itemsets
    assert len(frequent_itemsets) > 0
    
    # Check that we get results
    total_itemsets = sum(level.shape[0] for level in frequent_itemsets)
    assert total_itemsets > 0


def test_bitvector_vs_regular_apriori():
    """Test that bit vector and regular Apriori produce similar results"""
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)
    
    min_support = 0.4
    
    regular_result = priors.apriori(transactions, min_support)
    bitvector_result = priors.apriori_bitvector(transactions, min_support)
    
    # Both should find some patterns
    assert len(regular_result) > 0
    assert len(bitvector_result) > 0
    
    # Count total itemsets
    regular_total = sum(level.shape[0] for level in regular_result)
    bitvector_total = sum(level.shape[0] for level in bitvector_result)
    
    print(f"Regular Apriori found: {regular_total} itemsets")
    print(f"Bit vector Apriori found: {bitvector_total} itemsets")
    
    # They should find the same number of frequent itemsets
    assert regular_total == bitvector_total


def test_bitvector_too_many_items():
    """Test that bit vector fails gracefully with >64 items"""
    # Create dataset with 65 items (should fail)
    transactions = np.zeros((10, 65), dtype=np.int32)
    transactions[0, :5] = 1  # First transaction has items 0-4
    
    min_support = 0.1
    
    with pytest.raises(ValueError, match="only supports ≤64 items"):
        priors.apriori_bitvector(transactions, min_support)


def test_bitvector_performance_hint():
    """Test that bit vector should be faster (at least not much slower)"""
    # Create a reasonably sized dataset within 64 items
    np.random.seed(42)
    transactions = np.zeros((1000, 20), dtype=np.int32)
    
    for i in range(1000):
        size = max(1, min(20, int(np.random.normal(5, 2))))
        items = np.random.choice(20, size, replace=False)
        transactions[i, items] = 1
    
    min_support = 0.05
    
    import time
    
    # Time regular Apriori
    start = time.time()
    regular_result = priors.apriori(transactions, min_support)
    regular_time = time.time() - start
    
    # Time bit vector Apriori
    start = time.time()
    bitvector_result = priors.apriori_bitvector(transactions, min_support)
    bitvector_time = time.time() - start
    
    print(f"Regular Apriori: {regular_time:.4f}s")
    print(f"Bit vector Apriori: {bitvector_time:.4f}s")
    print(f"Speedup: {regular_time/bitvector_time:.2f}x")
    
    # They should find the same results
    regular_total = sum(level.shape[0] for level in regular_result)
    bitvector_total = sum(level.shape[0] for level in bitvector_result)
    assert regular_total == bitvector_total
    
    # Bit vector should be faster (or at least not much slower)
    # Allow up to 2x slower due to overhead on small datasets
    assert bitvector_time <= regular_time * 2.0


if __name__ == "__main__":
    pytest.main([__file__])