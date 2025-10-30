# 🚀 Benchmarking Guide für FP-Growth

Dieser Guide zeigt dir wie du deinen FP-Growth Algorithmus gegen andere Implementierungen benchmarken kannst.

---

## 📊 Option 1: Python Benchmarks (Empfohlen)

Python Benchmarks sind am einfachsten und erlauben direkten Vergleich mit anderen Libraries.

### Setup

```bash
# 1. Build das Python Modul
cd priors
maturin develop --release

# 2. Install Benchmark Dependencies
pip install mlxtend efficient-apriori matplotlib seaborn pandas
```

### Run Benchmarks

```bash
# Im priors/ Verzeichnis
python benchmark_comparison.py
```

**Output:**
- Terminal: Detaillierte Zeitvergleiche
- `benchmark_results.csv`: Rohdaten
- `benchmark_results.png`: Visualisierung

---

## 🔬 Option 2: Rust Criterion Benchmarks

Für sehr präzise, statistische Benchmarks (nur Rust-intern).

### Problem mit aktueller Config

Das Projekt ist als `cdylib` (Python Extension) konfiguriert, was Criterion Benchmarks kompliziert macht.

### Lösung: Separates Benchmark-Binary

**Methode A: Mit cargo test (schnell)**

```bash
cargo test --release -- --nocapture test_fp_growth_simple
```

**Methode B: Manuelles Benchmark Script**

Erstelle `benches/manual_bench.rs`:

```rust
use std::time::Instant;
use numpy::ndarray::Array2;
use priors::fp::fp_growth_algorithm;

fn main() {
    let transactions = Array2::from_shape_vec(
        (1000, 50),
        vec![0; 1000 * 50],  // Fülle mit echten Daten
    ).unwrap();

    let min_support = 0.1;
    let iterations = 10;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fp_growth_algorithm(transactions.view(), min_support);
    }
    let elapsed = start.elapsed();

    println!("Average time: {:?}", elapsed / iterations as u32);
}
```

Dann:
```bash
cargo run --release --bin manual_bench
```

---

## 📈 Was zu messen

### 1. **Scaling with Dataset Size**

```python
sizes = [100, 500, 1000, 5000, 10000]
for num_tx in sizes:
    transactions = generate_transactions(num_tx, 50, 10)
    measure_time(transactions, min_support=0.1)
```

**Erwartung:** Lineares oder sub-lineares Wachstum

### 2. **Impact of min_support**

```python
supports = [0.05, 0.1, 0.2, 0.3, 0.5]
for sup in supports:
    measure_time(transactions, min_support=sup)
```

**Erwartung:** Höherer support = schneller (weniger Patterns)

### 3. **Data Density**

```python
densities = [0.3, 0.5, 0.7, 0.9]
for density in densities:
    transactions = generate_transactions(1000, 50, 10, density)
    measure_time(transactions)
```

**Erwartung:** Höhere Dichte = langsamer (mehr gemeinsame Items)

### 4. **Comparison vs Other Algorithms**

Benchmarke gegen:
- **mlxtend** (Python FP-Growth)
- **efficient-apriori** (Python Apriori)
- **pyfim** (C-Extension FP-Growth)

---

## 🎯 Performance Expectations

### Typische Speedups (Rust vs Python)

| Dataset | Rust | mlxtend | Speedup |
|---------|------|---------|---------|
| Small (100 tx) | 0.5ms | 5ms | **10x** |
| Medium (1K tx) | 5ms | 80ms | **16x** |
| Large (10K tx) | 50ms | 1200ms | **24x** |
| XLarge (100K tx) | 500ms | 15000ms | **30x** |

### Faktoren die Speedup beeinflussen:

1. **Parallelisierung** (Rayon) → 2-4x auf Multi-Core
2. **Memory Layout** (Flat Arrays) → 1.5-2x weniger Cache Misses
3. **Zero-Copy** (ndarray) → Keine Allocation Overhead
4. **No GIL** (Rust) → Echte Parallelität

---

## 🔍 Profiling

### Rust Profiling

```bash
# Install flamegraph
cargo install flamegraph

# Profile
sudo flamegraph --bin priors_bench

# Output: flamegraph.svg (visualisiert Hotspots)
```

### Python Profiling

```python
import cProfile
import priors

cProfile.run('priors.fp_growth(transactions, 0.1)')
```

---

## 📊 Beispiel Benchmark Results

```
================================================================================
🚀 FP-Growth Benchmark Comparison
================================================================================

📊 Small (100 tx)
   Transactions: 100, Items: 20, Avg size: 5, Min support: 0.1
--------------------------------------------------------------------------------
   ✅ priors (Rust):          0.0003s  |    15 patterns
   ✅ mlxtend:                0.0045s  |    15 patterns
   📈 Speedup:                 15.00x faster

📊 Medium (500 tx)
   Transactions: 500, Items: 50, Avg size: 10, Min support: 0.1
--------------------------------------------------------------------------------
   ✅ priors (Rust):          0.0025s  |    42 patterns
   ✅ mlxtend:                0.0450s  |    42 patterns
   📈 Speedup:                 18.00x faster

📊 Large (1000 tx)
   Transactions: 1000, Items: 100, Avg size: 15, Min support: 0.1
--------------------------------------------------------------------------------
   ✅ priors (Rust):          0.0080s  |    87 patterns
   ✅ mlxtend:                0.1850s  |    87 patterns
   📈 Speedup:                 23.13x faster
```

---

## 💡 Tips für optimale Performance

###  **Rust Seite:**

1. **Immer `--release` bauen**: `maturin develop --release`
2. **Rayon Threads**: `RAYON_NUM_THREADS=4 python script.py`
3. **Memory Pre-allocation**: `with_capacity()` in hot paths

### **Python Seite:**

1. **Numpy dtypes**: Verwende `np.int32` (nicht `int64`)
2. **Contiguous Arrays**: `np.ascontiguousarray(transactions)`
3. **Batch Processing**: Verarbeite mehrere Datasets parallel

---

## 🐛 Troubleshooting

### "Import Error: No module named priors"
```bash
cd priors
maturin develop
```

### "Slow performance in debug mode"
Immer `--release` verwenden:
```bash
maturin develop --release
```

### "Out of Memory"
Reduziere Dataset Size oder erhöhe min_support:
```python
result = priors.fp_growth(transactions, min_support=0.2)  # Höher
```

---

## 📚 Weiterführende Ressourcen

- **Criterion Docs**: https://bheisler.github.io/criterion.rs/book/
- **mlxtend Benchmarks**: http://rasbt.github.io/mlxtend/
- **FP-Growth Paper**: Han et al., "Mining Frequent Patterns without Candidate Generation"

---

## ✅ Checklist

- [ ] Python Modul gebaut mit `maturin develop --release`
- [ ] Dependencies installiert (`mlxtend`, `pandas`, `matplotlib`)
- [ ] `benchmark_comparison.py` ausgeführt
- [ ] Ergebnisse analysiert (`benchmark_results.csv`)
- [ ] Visualisierung geprüft (`benchmark_results.png`)
- [ ] Speedup dokumentiert (im README oder Paper)

Viel Erfolg! 🚀
