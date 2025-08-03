import numpy as np
import pytest
from priors import apriori, calculate_support, calculate_confidence


def test_basic_apriori():
    """Test basic Apriori algorithm functionality"""
    # Simple transaction matrix
    # Items: [0, 1, 2, 3, 4]  
    # Transactions:
    # T1: {0, 1, 3}
    # T2: {0, 2, 3}  
    # T3: {1, 2, 3}
    # T4: {0, 1, 2}
    # T5: {0, 1, 3}
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
    ], dtype=np.int32)
    
    min_support = 0.4  # 2/5 = 0.4
    
    frequent_itemsets = apriori(transactions, min_support)
    
    # Should have at least 1-itemsets
    assert len(frequent_itemsets) > 0
    
    # Check that we get some frequent 1-itemsets
    level_1 = frequent_itemsets[0]
    assert level_1.shape[1] == 1  # 1-itemsets should have 1 column
    assert level_1.shape[0] > 0   # Should have some frequent items
    
    # Items 0, 1, 3 should be frequent (appear in 3+ transactions each)
    frequent_items = set(level_1.flatten())
    assert 0 in frequent_items  # Item 0 appears in T1, T2, T4, T5 (4 times)
    assert 1 in frequent_items  # Item 1 appears in T1, T3, T4, T5 (4 times) 
    assert 3 in frequent_items  # Item 3 appears in T1, T2, T3, T5 (4 times)


def test_support_calculation():
    """Test support calculation for specific itemsets"""
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
    ], dtype=np.int32)
    
    # Test single item support
    itemsets = np.array([[0], [1], [2], [3], [4]], dtype=np.uintp)
    support_values = calculate_support(transactions, itemsets)
    
    # Item 0 appears in T1, T2, T4, T5 = 4/5 = 0.8
    assert abs(support_values[0] - 0.8) < 1e-10
    
    # Item 1 appears in T1, T3, T4, T5 = 4/5 = 0.8  
    assert abs(support_values[1] - 0.8) < 1e-10
    
    # Item 2 appears in T2, T3, T4 = 3/5 = 0.6
    assert abs(support_values[2] - 0.6) < 1e-10
    
    # Item 3 appears in T1, T2, T3, T5 = 4/5 = 0.8
    assert abs(support_values[3] - 0.8) < 1e-10
    
    # Item 4 appears in none = 0/5 = 0.0
    assert abs(support_values[4] - 0.0) < 1e-10


def test_itemset_support():
    """Test support calculation for multi-item itemsets"""
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
    ], dtype=np.int32)
    
    # Test 2-itemset support
    itemsets = np.array([[0, 1], [0, 3], [1, 3]], dtype=np.uintp)
    support_values = calculate_support(transactions, itemsets)
    
    # {0, 1} appears in T1, T4, T5 = 3/5 = 0.6
    assert abs(support_values[0] - 0.6) < 1e-10
    
    # {0, 3} appears in T1, T2, T5 = 3/5 = 0.6
    assert abs(support_values[1] - 0.6) < 1e-10
    
    # {1, 3} appears in T1, T3, T5 = 3/5 = 0.6
    assert abs(support_values[2] - 0.6) < 1e-10


def test_confidence_calculation():
    """Test confidence calculation for association rules"""
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
    ], dtype=np.int32)
    
    # Test rule: {0} -> {1}
    # Support({0}) = 4/5 = 0.8
    # Support({0, 1}) = 3/5 = 0.6
    # Confidence = 0.6 / 0.8 = 0.75
    antecedent = np.array([0], dtype=np.uintp)
    consequent = np.array([1], dtype=np.uintp)
    confidence = calculate_confidence(transactions, antecedent, consequent)
    assert abs(confidence - 0.75) < 1e-10
    
    # Test rule: {1} -> {3}
    # Support({1}) = 4/5 = 0.8
    # Support({1, 3}) = 3/5 = 0.6  
    # Confidence = 0.6 / 0.8 = 0.75
    antecedent = np.array([1], dtype=np.uintp)
    consequent = np.array([3], dtype=np.uintp)
    confidence = calculate_confidence(transactions, antecedent, consequent)
    assert abs(confidence - 0.75) < 1e-10
    
    # Test rule: {2} -> {3}
    # Support({2}) = 3/5 = 0.6
    # Support({2, 3}) = 2/5 = 0.4 (T2, T3)
    # Confidence = 0.4 / 0.6 = 2/3 ≈ 0.6667
    antecedent = np.array([2], dtype=np.uintp)
    consequent = np.array([3], dtype=np.uintp)
    confidence = calculate_confidence(transactions, antecedent, consequent)
    assert abs(confidence - (2.0/3.0)) < 1e-10


def test_empty_transactions():
    """Test behavior with empty transaction matrix"""
    transactions = np.array([], dtype=np.int32).reshape(0, 5)
    min_support = 0.1
    
    frequent_itemsets = apriori(transactions, min_support)
    
    # Should return empty list for empty transactions
    assert len(frequent_itemsets) == 0


def test_high_min_support():
    """Test behavior with very high minimum support"""
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)
    
    min_support = 0.9  # Very high support threshold
    
    frequent_itemsets = apriori(transactions, min_support)
    
    # Should return empty or very few itemsets
    assert len(frequent_itemsets) <= 1


def test_single_transaction():
    """Test with single transaction"""
    transactions = np.array([[1, 0, 1, 1, 0]], dtype=np.int32)
    min_support = 0.5  # 50% support
    
    frequent_itemsets = apriori(transactions, min_support)
    
    # With single transaction, any present item has 100% support
    assert len(frequent_itemsets) > 0
    level_1 = frequent_itemsets[0]
    frequent_items = set(level_1.flatten())
    
    # Items 0, 2, 3 should be present (have 100% support)
    expected_items = {0, 2, 3}
    assert frequent_items == expected_items


if __name__ == "__main__":
    pytest.main([__file__])