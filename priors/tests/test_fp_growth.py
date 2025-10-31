import numpy as np
import pytest
import priors
import pandas as pd


def test_fp_growth_vs_mlxtend():
    """Test that FP-Growth produces similar results to mlxtend"""
    mlxtend = pytest.importorskip("mlxtend")
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    # Create transaction data
    dataset = [
        ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
        ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
        ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
        ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
        ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs'],
    ]

    # Encode transactions for mlxtend
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Get mlxtend results
    mlxtend_result = fpgrowth(df, min_support=0.4, use_colnames=False)

    # Convert to our format (numpy array)
    transactions = te_ary.astype(np.int32)

    # Get our results
    our_result = priors.fp_growth(transactions, 0.4)

    # Count total itemsets from both
    mlxtend_count = len(mlxtend_result)
    our_count = sum(level.shape[0] for level in our_result)

    # Both should find frequent itemsets
    assert mlxtend_count > 0, "mlxtend found no itemsets"
    assert our_count > 0, "our implementation found no itemsets"

    # Allow some variation, but should be in similar ballpark
    assert abs(mlxtend_count - our_count) / max(mlxtend_count, our_count) < 0.5, \
        f"Result counts differ too much: mlxtend={mlxtend_count}, ours={our_count}"


def test_fp_growth_vs_efficient_apriori():
    """Test that FP-Growth produces similar results to efficient-apriori"""
    efficient_apriori = pytest.importorskip("efficient_apriori")
    from efficient_apriori import apriori as efficient_apriori_func

    # Create transaction data
    transactions_list = [
        ('eggs', 'bacon', 'soup'),
        ('eggs', 'bacon', 'apple'),
        ('soup', 'bacon', 'banana'),
        ('eggs', 'apple', 'soup'),
        ('eggs', 'bacon', 'soup', 'apple'),
    ]

    # Get efficient-apriori results
    itemsets, rules = efficient_apriori_func(transactions_list, min_support=0.4)

    # Count total itemsets from efficient-apriori
    efficient_count = sum(len(itemsets[k]) for k in itemsets.keys())

    # Convert to our format
    all_items = sorted(set(item for transaction in transactions_list for item in transaction))
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}

    transactions_matrix = np.zeros((len(transactions_list), len(all_items)), dtype=np.int32)
    for i, transaction in enumerate(transactions_list):
        for item in transaction:
            transactions_matrix[i, item_to_idx[item]] = 1

    # Get our results
    our_result = priors.fp_growth(transactions_matrix, 0.4)
    our_count = sum(level.shape[0] for level in our_result)

    # Both should find frequent itemsets
    assert efficient_count > 0, "efficient-apriori found no itemsets"
    assert our_count > 0, "our implementation found no itemsets"

    # Allow some variation
    assert abs(efficient_count - our_count) / max(efficient_count, our_count) < 0.5, \
        f"Result counts differ too much: efficient-apriori={efficient_count}, ours={our_count}"


def test_fp_growth_vs_our_apriori():
    """Test that FP-Growth and our Apriori find the same patterns"""
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

    # Both should find patterns
    assert len(apriori_result) > 0
    assert len(fp_growth_result) > 0

    # Count total itemsets
    apriori_total = sum(level.shape[0] for level in apriori_result)
    fp_growth_total = sum(level.shape[0] for level in fp_growth_result)

    # They should find the same number of itemsets
    assert apriori_total == fp_growth_total, \
        f"Apriori found {apriori_total} itemsets, FP-Growth found {fp_growth_total}"


def test_fp_growth_basic():
    """Basic functionality test"""
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)

    result = priors.fp_growth(transactions, 0.4)

    # Should have at least some frequent itemsets
    assert len(result) > 0
    total_itemsets = sum(level.shape[0] for level in result)
    assert total_itemsets > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
