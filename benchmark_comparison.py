#!/usr/bin/env python3
"""
Benchmark Comparison: priors (Rust) vs mlxtend (Python) vs efficient-apriori
"""

import time
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries to compare
try:
    import priors  # Your Rust implementation
    HAS_PRIORS = True
except ImportError:
    HAS_PRIORS = False
    print("⚠️  priors not installed. Run: maturin develop")

try:
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("⚠️  mlxtend not installed. Run: pip install mlxtend")

try:
    from efficient_apriori import apriori as efficient_apriori
    HAS_EFFICIENT_APRIORI = True
except ImportError:
    HAS_EFFICIENT_APRIORI = False
    print("⚠️  efficient-apriori not installed. Run: pip install efficient-apriori")


def generate_transactions(num_transactions: int, num_items: int,
                         avg_items_per_tx: int, density: float = 0.7) -> np.ndarray:
    """Generate synthetic transaction data."""
    np.random.seed(42)
    data = np.zeros((num_transactions, num_items), dtype=np.int32)

    for i in range(num_transactions):
        num_items_in_tx = int(avg_items_per_tx * (0.5 + np.random.random()))
        num_items_in_tx = min(num_items_in_tx, num_items)

        if np.random.random() < density:
            items = np.random.choice(num_items, size=num_items_in_tx, replace=False)
            data[i, items] = 1

    return data


def transactions_to_list(transactions: np.ndarray) -> List[List[int]]:
    """Convert binary matrix to list of transactions."""
    return [list(np.where(row == 1)[0]) for row in transactions]


def benchmark_priors(transactions: np.ndarray, min_support: float) -> Tuple[float, int]:
    """Benchmark priors (Rust FP-Growth)."""
    start = time.time()
    result = priors.fp_growth(transactions, min_support)
    elapsed = time.time() - start

    # Count total patterns
    total_patterns = sum(len(level) for level in result)
    return elapsed, total_patterns


def benchmark_mlxtend(transactions: np.ndarray, min_support: float) -> Tuple[float, int]:
    """Benchmark mlxtend FP-Growth."""
    # Convert to DataFrame format
    df = pd.DataFrame(transactions.astype(bool), columns=[f"item_{i}" for i in range(transactions.shape[1])])

    start = time.time()
    frequent_itemsets = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=True)
    elapsed = time.time() - start

    return elapsed, len(frequent_itemsets)


def benchmark_efficient_apriori(transactions: np.ndarray, min_support: float) -> Tuple[float, int]:
    """Benchmark efficient-apriori."""
    tx_list = transactions_to_list(transactions)
    min_support_count = int(min_support * len(tx_list))

    start = time.time()
    itemsets, rules = efficient_apriori(tx_list, min_support=min_support_count/len(tx_list))
    elapsed = time.time() - start

    total_patterns = sum(len(itemsets[k]) for k in itemsets)
    return elapsed, total_patterns


def run_comparison(configs: List[dict]):
    """Run comprehensive comparison."""
    results = []

    print("\n" + "="*80)
    print("🚀 FP-Growth Benchmark Comparison")
    print("="*80 + "\n")

    for config in configs:
        name = config['name']
        num_tx = config['num_transactions']
        num_items = config['num_items']
        avg_size = config['avg_size']
        min_sup = config['min_support']

        print(f"📊 {name}")
        print(f"   Transactions: {num_tx}, Items: {num_items}, Avg size: {avg_size}, Min support: {min_sup}")
        print("-" * 80)

        # Generate data
        transactions = generate_transactions(num_tx, num_items, avg_size)

        # Benchmark each implementation
        if HAS_PRIORS:
            try:
                time_priors, patterns_priors = benchmark_priors(transactions, min_sup)
                print(f"   ✅ priors (Rust):         {time_priors:8.4f}s  |  {patterns_priors:6d} patterns")
                results.append({
                    'config': name,
                    'library': 'priors (Rust)',
                    'time': time_priors,
                    'patterns': patterns_priors
                })
            except Exception as e:
                print(f"   ❌ priors (Rust):         ERROR - {e}")

        if HAS_MLXTEND:
            try:
                time_mlxtend, patterns_mlxtend = benchmark_mlxtend(transactions, min_sup)
                print(f"   ✅ mlxtend:               {time_mlxtend:8.4f}s  |  {patterns_mlxtend:6d} patterns")
                results.append({
                    'config': name,
                    'library': 'mlxtend',
                    'time': time_mlxtend,
                    'patterns': patterns_mlxtend
                })

                if HAS_PRIORS:
                    speedup = time_mlxtend / time_priors
                    print(f"   📈 Speedup:                {speedup:7.2f}x faster")
            except Exception as e:
                print(f"   ❌ mlxtend:               ERROR - {e}")

        if HAS_EFFICIENT_APRIORI:
            try:
                time_ea, patterns_ea = benchmark_efficient_apriori(transactions, min_sup)
                print(f"   ✅ efficient-apriori:     {time_ea:8.4f}s  |  {patterns_ea:6d} patterns")
                results.append({
                    'config': name,
                    'library': 'efficient-apriori',
                    'time': time_ea,
                    'patterns': patterns_ea
                })
            except Exception as e:
                print(f"   ❌ efficient-apriori:     ERROR - {e}")

        print()

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame):
    """Plot benchmark results."""
    if df.empty:
        print("No results to plot")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Execution time comparison
    sns.barplot(data=df, x='config', y='time', hue='library', ax=ax1)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dataset Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_yscale('log')
    ax1.legend(title='Library')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Speedup vs mlxtend
    if 'mlxtend' in df['library'].values and 'priors (Rust)' in df['library'].values:
        speedup_data = []
        for config in df['config'].unique():
            priors_time = df[(df['config'] == config) & (df['library'] == 'priors (Rust)')]['time'].values
            mlxtend_time = df[(df['config'] == config) & (df['library'] == 'mlxtend')]['time'].values

            if len(priors_time) > 0 and len(mlxtend_time) > 0:
                speedup = mlxtend_time[0] / priors_time[0]
                speedup_data.append({'config': config, 'speedup': speedup})

        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            sns.barplot(data=speedup_df, x='config', y='speedup', ax=ax2, color='green', alpha=0.7)
            ax2.axhline(y=1, color='red', linestyle='--', label='No speedup')
            ax2.set_title('Speedup vs mlxtend', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Dataset Configuration')
            ax2.set_ylabel('Speedup Factor (x)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved to: benchmark_results.png")
    plt.show()


if __name__ == '__main__':
    # Define benchmark configurations
    configs = [
        {
            'name': 'Small (100 tx)',
            'num_transactions': 100,
            'num_items': 20,
            'avg_size': 5,
            'min_support': 0.1
        },
        {
            'name': 'Medium (500 tx)',
            'num_transactions': 500,
            'num_items': 50,
            'avg_size': 10,
            'min_support': 0.1
        },
        {
            'name': 'Large (1000 tx)',
            'num_transactions': 1000,
            'num_items': 100,
            'avg_size': 15,
            'min_support': 0.1
        },
        {
            'name': 'XLarge (5000 tx)',
            'num_transactions': 5000,
            'num_items': 100,
            'avg_size': 20,
            'min_support': 0.05
        },
    ]

    # Run benchmarks
    results_df = run_comparison(configs)

    # Save results
    results_df.to_csv('benchmark_results.csv', index=False)
    print(f"\n💾 Results saved to: benchmark_results.csv\n")

    # Display summary
    print("\n" + "="*80)
    print("📈 Summary Statistics")
    print("="*80)
    print(results_df.groupby('library')['time'].agg(['mean', 'std', 'min', 'max']))

    # Plot results
    try:
        plot_results(results_df)
    except Exception as e:
        print(f"Could not plot results: {e}")
