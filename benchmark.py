#!/usr/bin/env python3

import os
import time

import numpy as np
import psutil

try:
    import priors

    HAS_PRIORS = True
except ImportError:
    print("❌ priors not installed")
    exit(1)

try:
    import pandas as pd
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("⚠️  mlxtend not installed")

try:
    from efficient_apriori import apriori as efficient_apriori

    HAS_EFFICIENT_APRIORI = True
except ImportError:
    HAS_EFFICIENT_APRIORI = False
    print("⚠️  efficient-apriori not installed")


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def generate_data(num_tx, num_items, avg_size, density):
    print(f"  Generating {num_tx:,} × {num_items} transactions...", end="", flush=True)
    np.random.seed(42)
    data = np.zeros((num_tx, num_items), dtype=np.int32)

    for i in range(num_tx):
        size = int(avg_size * (0.5 + np.random.random()))
        if np.random.random() < density:
            items = np.random.choice(num_items, min(size, num_items), replace=False)
            data[i, items] = 1

    print(f" ✓ {data.nbytes / 1024 / 1024:.1f}MB")
    return data


def benchmark(name, func, data, min_sup):
    print(f"  {name:25s}", end="", flush=True)
    mem_start = get_memory_mb()

    try:
        start = time.time()
        result = func(data, min_sup)
        elapsed = time.time() - start
        mem_end = get_memory_mb()

        if isinstance(result, list):
            patterns = sum(len(itemset_list) for itemset_list in result)
        elif isinstance(result, pd.DataFrame):
            patterns = len(result)
        else:
            patterns = sum(len(result[0][k]) for k in result[0])

        print(
            f" {elapsed:6.2f}s | {patterns:8,} patterns | {mem_end - mem_start:6.1f}MB"
        )
        return elapsed, patterns, mem_end - mem_start
    except MemoryError:
        print(" ❌ OOM")
        return None, None, None
    except Exception as e:
        print(f" ❌ {str(e)[:50]}")
        return None, None, None


def test_priors_regular(data, sup):
    return priors.fp_growth(data, sup)


def test_priors_lazy(data, sup):
    pid = priors.create_lazy_fp_growth()
    chunk_size = 5000

    for i in range(0, data.shape[0], chunk_size):
        priors.lazy_count_pass(pid, data[i : i + chunk_size])

    priors.lazy_finalize_counts(pid, sup)

    for i in range(0, data.shape[0], chunk_size):
        priors.lazy_build_pass(pid, data[i : i + chunk_size])

    result = priors.lazy_mine_patterns(pid, sup)
    priors.lazy_cleanup(pid)
    return result


def test_mlxtend(data, sup):
    df = pd.DataFrame(
        data.astype(bool), columns=[f"i{i}" for i in range(data.shape[1])]
    )
    return mlxtend_fpgrowth(df, min_support=sup, use_colnames=True)


def test_efficient_apriori(data, sup):
    tx_list = [list(np.where(row == 1)[0]) for row in data]
    return efficient_apriori(tx_list, min_support=sup)


configs = [
    {
        "name": "10K × 50",
        "tx": 10_000,
        "items": 50,
        "size": 20,
        "dens": 0.7,
        "sup": 0.02,
    },
    {
        "name": "30K × 80",
        "tx": 30_000,
        "items": 80,
        "size": 35,
        "dens": 0.75,
        "sup": 0.01,
    },
    {
        "name": "60K × 100",
        "tx": 60_000,
        "items": 100,
        "size": 50,
        "dens": 0.8,
        "sup": 0.008,
    },
    {
        "name": "100K × 120",
        "tx": 100_000,
        "items": 120,
        "size": 60,
        "dens": 0.85,
        "sup": 0.005,
    },
]

print("\n" + "=" * 80)
print("⚡ FP-Growth Benchmark")
print("=" * 80)
print(
    f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB | Available: {psutil.virtual_memory().available / 1024**3:.1f}GB"
)
print("=" * 80)

results = []

for cfg in configs:
    print(f"\n📊 {cfg['name']} (density={cfg['dens']}, support={cfg['sup']})")
    print("-" * 80)

    data = generate_data(cfg["tx"], cfg["items"], cfg["size"], cfg["dens"])

    t1, p1, m1 = benchmark("priors (regular)", test_priors_regular, data, cfg["sup"])
    t2, p2, m2 = benchmark("priors (lazy)", test_priors_lazy, data, cfg["sup"])

    if HAS_MLXTEND:
        t3, p3, m3 = benchmark("mlxtend", test_mlxtend, data, cfg["sup"])
    else:
        t3, p3, m3 = None, None, None

    if HAS_EFFICIENT_APRIORI:
        t4, p4, m4 = benchmark(
            "efficient-apriori", test_efficient_apriori, data, cfg["sup"]
        )
    else:
        t4, p4, m4 = None, None, None

    if t1 and t2:
        print(
            f"\n  💡 Lazy vs Regular: {((t2 / t1 - 1) * 100):+.1f}% time | {((1 - m2 / m1) * 100):+.1f}% memory savings"
        )

    if t1 and t3:
        print(f"  💡 priors vs mlxtend: {(t3 / t1):.1f}x faster")

    if t1 and t4:
        print(f"  💡 priors vs efficient-apriori: {(t4 / t1):.1f}x faster")

    results.append(
        {
            "dataset": cfg["name"],
            "priors_time": t1,
            "lazy_time": t2,
            "mlxtend_time": t3,
            "efficient_time": t4,
            "patterns": p1,
        }
    )

print("\n" + "=" * 80)
print("📈 Summary")
print("=" * 80)

for r in results:
    print(f"\n{r['dataset']}:")
    print(f"  Patterns: {r['patterns']:,}" if r["patterns"] else "  Patterns: N/A")
    if r["priors_time"]:
        print(f"  priors (regular): {r['priors_time']:.3f}s")
    if r["lazy_time"]:
        print(f"  priors (lazy):    {r['lazy_time']:.3f}s")
    if r["mlxtend_time"]:
        print(
            f"  mlxtend:          {r['mlxtend_time']:.3f}s ({r['mlxtend_time'] / r['priors_time']:.1f}x slower)"
            if r["priors_time"]
            else f"  mlxtend:          {r['mlxtend_time']:.3f}s"
        )
    if r["efficient_time"]:
        print(
            f"  efficient-apriori: {r['efficient_time']:.3f}s ({r['efficient_time'] / r['priors_time']:.1f}x slower)"
            if r["priors_time"]
            else f"  efficient-apriori: {r['efficient_time']:.3f}s"
        )

print("\n" + "=" * 80)
print("✓ Benchmark Complete")
print("=" * 80)
