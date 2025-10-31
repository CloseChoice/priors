import numpy as np
import pytest
import priors
import pandas as pd


def test_fp_growth_vs_mlxtend():
    """Test that FP-Growth produces similar results to mlxtend"""
    mlxtend = pytest.importorskip("mlxtend")
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    dataset = [
        ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
        ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
        ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
        ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
        ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs'],
    ]

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    mlxtend_result = fpgrowth(df, min_support=0.4, use_colnames=False)
    transactions = te_ary.astype(np.int32)
    our_result = priors.fp_growth(transactions, 0.4)

    mlxtend_count = len(mlxtend_result)
    our_count = sum(level.shape[0] for level in our_result)

    assert mlxtend_count > 0, "MLxtend should find itemsets"
    assert our_count > 0, "Priors should find itemsets"
    # Both should find exactly the same number of frequent itemsets
    assert mlxtend_count == our_count, f"Count mismatch: priors={our_count}, mlxtend={mlxtend_count}"


def test_fp_growth_vs_efficient_apriori():
    """Test that FP-Growth produces similar results to efficient-apriori"""
    efficient_apriori = pytest.importorskip("efficient_apriori")
    from efficient_apriori import apriori as efficient_apriori_func

    transactions_list = [
        ('eggs', 'bacon', 'soup'),
        ('eggs', 'bacon', 'apple'),
        ('soup', 'bacon', 'banana'),
        ('eggs', 'apple', 'soup'),
        ('eggs', 'bacon', 'soup', 'apple'),
    ]

    itemsets, rules = efficient_apriori_func(transactions_list, min_support=0.4)
    efficient_count = sum(len(itemsets[k]) for k in itemsets.keys())

    all_items = sorted(set(item for transaction in transactions_list for item in transaction))
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}

    transactions_matrix = np.zeros((len(transactions_list), len(all_items)), dtype=np.int32)
    for i, transaction in enumerate(transactions_list):
        for item in transaction:
            transactions_matrix[i, item_to_idx[item]] = 1

    our_result = priors.fp_growth(transactions_matrix, 0.4)
    our_count = sum(level.shape[0] for level in our_result)

    assert efficient_count > 0, "Efficient-apriori should find itemsets"
    assert our_count > 0, "Priors should find itemsets"
    # Both should find exactly the same number of frequent itemsets
    assert efficient_count == our_count, f"Count mismatch: priors={our_count}, efficient-apriori={efficient_count}"


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

    assert len(result) > 0
    total_itemsets = sum(level.shape[0] for level in result)
    assert total_itemsets > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
