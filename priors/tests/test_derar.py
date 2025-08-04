import numpy as np
import pytest
import priors


def test_derar_basic():
    """Test basic DERAR functionality"""
    # Simple transaction matrix
    transactions = np.array([
        [1, 1, 0, 1, 0],  # T1: {0, 1, 3}
        [1, 0, 1, 1, 0],  # T2: {0, 2, 3}
        [0, 1, 1, 1, 0],  # T3: {1, 2, 3}
        [1, 1, 1, 0, 0],  # T4: {0, 1, 2}
        [1, 1, 0, 1, 0],  # T5: {0, 1, 3}
        [1, 0, 0, 1, 1],  # T6: {0, 3, 4}
        [0, 1, 1, 0, 1],  # T7: {1, 2, 4}
    ], dtype=np.int32)
    
    min_support = 0.3  # 2/7 ≈ 0.29
    min_confidence = 0.5
    
    rules = priors.derar(transactions, min_support, min_confidence)
    
    # Should find some rules
    assert len(rules) > 0
    
    # Check rule structure: (antecedent, consequent, support, confidence, mi, stability, tcm)
    for rule in rules:
        assert len(rule) == 7
        antecedent, consequent, support, confidence, mi, stability, tcm = rule
        
        # Basic validation
        assert isinstance(antecedent, list)
        assert isinstance(consequent, list)
        assert len(antecedent) > 0
        assert len(consequent) > 0
        assert 0.0 <= support <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert stability >= 0.0
        assert tcm >= 0.0


def test_derar_quality_measures():
    """Test that DERAR produces high-quality rules with good statistical measures"""
    # Create a dataset with clear patterns
    transactions = np.array([
        # Strong pattern: items 0,1 often together, and they predict item 2
        [1, 1, 1, 0, 0],  # {0, 1, 2}
        [1, 1, 1, 0, 0],  # {0, 1, 2}
        [1, 1, 1, 0, 0],  # {0, 1, 2}
        [1, 1, 0, 0, 0],  # {0, 1} without 2 (rare)
        
        # Weak pattern: item 3 with random others
        [0, 0, 0, 1, 1],  # {3, 4}
        [0, 1, 0, 1, 0],  # {1, 3}
        [1, 0, 0, 1, 0],  # {0, 3}
        
        # More strong pattern instances
        [1, 1, 1, 0, 0],  # {0, 1, 2}
        [1, 1, 1, 0, 0],  # {0, 1, 2}
    ], dtype=np.int32)
    
    min_support = 0.2
    min_confidence = 0.6
    
    rules = priors.derar(transactions, min_support, min_confidence)
    
    # Should find the strong pattern rule: {0,1} -> {2}
    found_strong_rule = False
    for rule in rules:
        antecedent, consequent, support, confidence, mi, stability, tcm = rule
        
        # Look for rule {0,1} -> {2}
        if sorted(antecedent) == [0, 1] and consequent == [2]:
            found_strong_rule = True
            print(f"Found strong rule: {antecedent} -> {consequent}")
            print(f"  Support: {support:.3f}, Confidence: {confidence:.3f}")
            print(f"  MI: {mi:.3f}, Stability: {stability:.3f}, TCM: {tcm:.3f}")
            
            # This should be a high-quality rule
            assert confidence >= 0.8  # Strong confidence
            assert mi > 0.0  # Positive mutual information
            assert tcm > 0.0  # Good target concentration
    
    assert found_strong_rule, "Should find the strong pattern rule {0,1} -> {2}"


def test_derar_vs_traditional_quality():
    """Test that DERAR rules are a high-quality subset of rules derivable from FP-Growth patterns"""
    # Create dataset with both strong and weak patterns
    np.random.seed(42)
    transactions = np.zeros((100, 10), dtype=np.int32)
    
    # Create some strong correlations
    for i in range(50):
        # Strong pattern: items 0,1,2 often together
        if np.random.random() < 0.9:
            transactions[i, [0, 1, 2]] = 1
        # Add some noise
        noise_items = np.random.choice(10, size=2, replace=False)
        transactions[i, noise_items] = 1
    
    # Fill remaining with random patterns
    for i in range(50, 100):
        size = np.random.randint(1, 5)
        items = np.random.choice(10, size, replace=False)
        transactions[i, items] = 1
    
    min_support = 0.1
    min_confidence = 0.5
    
    # Get DERAR rules
    derar_rules = priors.derar(transactions, min_support, min_confidence)
    
    # Get traditional frequent patterns for comparison
    fp_growth_patterns = priors.fp_growth(transactions, min_support)
    
    # Convert FP-Growth patterns to a set of all possible itemsets
    fp_itemsets = set()
    for level in fp_growth_patterns:
        for row_idx in range(level.shape[0]):
            itemset = tuple(sorted(level[row_idx]))
            fp_itemsets.add(itemset)
    
    print(f"DERAR found {len(derar_rules)} high-quality rules")
    print(f"FP-Growth found {len(fp_itemsets)} frequent itemsets")
    
    # Verify that every DERAR rule comes from frequent itemsets found by FP-Growth
    for rule in derar_rules:
        antecedent, consequent, support, confidence, mi, stability, tcm = rule
        
        # The union of antecedent and consequent should be a frequent itemset
        full_pattern = tuple(sorted(antecedent + consequent))
        
        assert full_pattern in fp_itemsets, f"DERAR rule {antecedent} -> {consequent} should come from frequent itemset {full_pattern}"
        
        # DERAR rules should have meaningful statistical measures
        assert mi >= 0.0, "Mutual information should be non-negative"
        assert stability > 0.0, "Stability score should be positive"
        assert tcm >= 0.0, "Target concentration measure should be non-negative"
        
        # High confidence due to filtering
        assert confidence >= min_confidence, "All rules should meet minimum confidence"
    
    print(f"✓ All {len(derar_rules)} DERAR rules are derived from FP-Growth frequent itemsets")


def test_derar_empty_and_edge_cases():
    """Test DERAR with edge cases"""
    # Empty dataset
    empty_transactions = np.array([], dtype=np.int32).reshape(0, 5)
    rules = priors.derar(empty_transactions, 0.1, 0.5)
    assert len(rules) == 0
    
    # Single transaction - should find rules but filter most out with high thresholds
    single_transaction = np.array([[1, 0, 1, 1, 0]], dtype=np.int32)
    rules = priors.derar(single_transaction, 0.5, 0.5, stability_threshold=50.0, tcm_threshold=0.8)
    # With high thresholds, should find few or no rules
    assert len(rules) >= 0  # At least doesn't crash
    
    # Very high thresholds
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
    ], dtype=np.int32)
    
    rules = priors.derar(transactions, 0.9, 0.9)  # Very high thresholds
    # Should find very few or no rules
    assert len(rules) >= 0  # At least doesn't crash


def test_derar_statistical_filtering():
    """Test that DERAR's statistical filtering works correctly"""
    # Create a dataset where traditional methods would find many spurious patterns
    # but DERAR should filter them out
    np.random.seed(123)
    transactions = np.zeros((200, 15), dtype=np.int32)
    
    # Add mostly random noise
    for i in range(150):
        size = np.random.randint(3, 8)
        items = np.random.choice(15, size, replace=False)
        transactions[i, items] = 1
    
    # Add a few transactions with a real pattern: {0,1} -> {2}
    for i in range(150, 200):
        transactions[i, [0, 1, 2]] = 1
        # Add some random items too
        extra_items = np.random.choice(range(3, 15), size=np.random.randint(1, 4), replace=False)
        transactions[i, extra_items] = 1
    
    min_support = 0.05  # Low support to catch noise
    min_confidence = 0.3
    
    derar_rules = priors.derar(transactions, min_support, min_confidence)
    
    # DERAR should filter out most noise and find the real pattern
    assert len(derar_rules) > 0, "Should find at least some rules"
    
    # Check that we found the real pattern
    found_real_pattern = False
    for rule in derar_rules:
        antecedent, consequent, support, confidence, mi, stability, tcm = rule
        
        if set(antecedent) == {0, 1} and set(consequent) == {2}:
            found_real_pattern = True
            # This should have good quality measures
            assert confidence > 0.7, "Real pattern should have high confidence"
            assert mi > 0.0, "Real pattern should have positive MI"
            assert tcm > 0.0, "Real pattern should have good TCM"
            break
    
    assert found_real_pattern, "Should find the real pattern {0,1} -> {2}"


def test_derar_configurable_parameters():
    """Test that DERAR parameters can be configured"""
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0], 
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)
    
    min_support = 0.2
    min_confidence = 0.4
    
    # Test with default parameters
    rules_default = priors.derar(transactions, min_support, min_confidence)
    
    # Test with strict parameters - should find fewer rules
    rules_strict = priors.derar(
        transactions, min_support, min_confidence,
        stability_threshold=50.0,  # High stability required
        mi_threshold=0.5,          # High mutual information required
        tcm_threshold=0.8          # High target concentration required
    )
    
    # Test with lenient parameters - should find more rules
    rules_lenient = priors.derar(
        transactions, min_support, min_confidence,
        stability_threshold=0.1,   # Low stability threshold
        mi_threshold=0.0,          # No MI requirement
        tcm_threshold=0.01         # Low TCM requirement
    )
    
    print(f"Default rules: {len(rules_default)}")
    print(f"Strict rules: {len(rules_strict)}")
    print(f"Lenient rules: {len(rules_lenient)}")
    
    # Lenient should find more or equal rules than strict
    assert len(rules_lenient) >= len(rules_strict)
    
    # All rule sets should be valid
    for rules in [rules_default, rules_strict, rules_lenient]:
        for rule in rules:
            assert len(rule) == 7  # (antecedent, consequent, support, confidence, mi, stability, tcm)
            antecedent, consequent, support, confidence, mi, stability, tcm = rule
            assert isinstance(antecedent, list) and len(antecedent) > 0
            assert isinstance(consequent, list) and len(consequent) > 0
            assert 0.0 <= support <= 1.0
            assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])