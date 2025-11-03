# 🚀 Priors - High-Performance FP-Growth in Rust

Blazing-fast implementation of the FP-Growth algorithm for frequent pattern mining, written in Rust with Python bindings.

## ✨ Features

- **🔥 Fast**: 10-30x faster than Python implementations (mlxtend, efficient-apriori)
- **⚡ Parallel**: Multi-threaded mining using Rayon
- **💾 Memory-Efficient**: Flat array storage, minimal allocations
- **🐍 Python Integration**: Seamless numpy integration via PyO3
- **🧪 Well-Tested**: Comprehensive unit and integration tests
- **📊 Benchmarked**: Detailed performance comparisons available

## 🎯 What is FP-Growth?

FP-Growth (Frequent Pattern Growth) is an algorithm for mining frequent itemsets without candidate generation. It's faster than Apriori for dense datasets because it:

1. Builds a compact FP-Tree (prefix tree) to compress transactions
2. Uses divide-and-conquer to mine patterns recursively
3. Avoids generating candidate itemsets explicitly

**Use Cases:**
- Market basket analysis (shopping patterns)
- Web usage mining (click patterns)
- Bioinformatics (gene expression patterns)
- Text mining (word co-occurrence)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd priors

# Build and install Python package
cd priors
maturin develop --release
```

### Usage

```python
import numpy as np
import priors

# Create transaction matrix (rows=transactions, cols=items)
transactions = np.array([
    [1, 1, 0, 0, 1],  # Transaction 1: items A, B, E
    [1, 1, 1, 0, 0],  # Transaction 2: items A, B, C
    [1, 0, 1, 1, 0],  # Transaction 3: items A, C, D
    [0, 1, 1, 0, 0],  # Transaction 4: items B, C
], dtype=np.int32)

# Mine frequent itemsets with 50% minimum support
result = priors.fp_growth(transactions, min_support=0.5)

# result[0]: All frequent 1-itemsets
# result[1]: All frequent 2-itemsets
# result[2]: All frequent 3-itemsets, etc.

for level_idx, itemsets in enumerate(result):
    print(f"Frequent {level_idx + 1}-itemsets: {len(itemsets)}")
    print(itemsets)
```

## 📊 Performance

Benchmark on 1000 transactions, 50 items, 10 avg items/transaction:

| Implementation | Time | Speedup |
|---------------|------|---------|
| **priors (Rust)** | **8ms** | **23x** |
| mlxtend (Python) | 185ms | 1x |
| efficient-apriori | 220ms | 0.84x |

See [BENCHMARKING.md](BENCHMARKING.md) for detailed benchmarks and how to run them.

## 🏗️ Project Structure

```
priors/
├── src/
│   ├── lib.rs              # Python bindings
│   └── fp/                 # FP-Growth module
│       ├── mod.rs          # Module exports
│       ├── storage.rs      # Memory-efficient storage
│       ├── tree/           # FP-Tree implementation
│       │   ├── mod.rs
│       │   ├── tree.rs     # Tree structures
│       │   └── tree_ops.rs # Tree operations
│       ├── builder.rs      # Tree construction
│       ├── mining.rs       # Main algorithm
│       ├── combinations.rs # Single-path optimization
│       └── tests.rs        # Unit tests
├── benches/               # Benchmarks
├── Cargo.toml            # Rust dependencies
└── pyproject.toml        # Python package config
```

## 🧪 Testing

```bash
# Run Rust tests
cargo test

# Run with output
cargo test -- --nocapture

# Run benchmarks
python benchmark_comparison.py
```

## 🎓 Learning Resources

### Understanding the Algorithm

1. **Start with `storage.rs`**: Learn about flat array storage
2. **Then `tree/tree.rs`**: Understand FP-Tree structure
3. **Then `tree/tree_ops.rs`**: See how the tree is manipulated
4. **Then `builder.rs`**: Learn tree construction
5. **Finally `mining.rs`**: The main algorithm

### Key Concepts

**FP-Tree Structure:**
```
Transactions: [[A,B,C], [A,B,D], [A,C]]

FP-Tree:  root → A:3 → B:2 → C:1
                      └→ D:1
                  └→ C:1
```
Common prefixes are shared, saving memory!

**Conditional Pattern Base:**
For item C, get all paths leading to C:
- Path 1: `[A, B]` with count 1
- Path 2: `[A]` with count 1

Build conditional tree for C and recurse!

## 🔧 Development

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- Python 3.7+
- maturin (`pip install maturin`)

### Build

```bash
# Development build (fast compile, slow runtime)
maturin develop

# Release build (slow compile, fast runtime)
maturin develop --release

# Just compile Rust (no Python)
cargo build --release
```

### Adding Features

1. **New storage format**: Edit `storage.rs`
2. **Tree optimizations**: Edit `tree/tree_ops.rs`
3. **Algorithm variants**: Edit `mining.rs`
4. **Python API**: Edit `lib.rs`

## 📈 Optimization Tips

### Rust Side

- Use `with_capacity()` for pre-allocation
- Profile with `cargo flamegraph`
- Tune Rayon threads: `RAYON_NUM_THREADS=4`

### Python Side

- Use `np.int32` (not `int64` or Python int)
- Ensure contiguous arrays: `np.ascontiguousarray()`
- Batch process multiple datasets

## 🤝 Contributing

Contributions welcome! Areas to improve:

- [ ] More algorithm variants (Apriori, Eclat)
- [ ] Streaming FP-Growth for large datasets
- [ ] GPU acceleration (CUDA)
- [ ] More Python examples
- [ ] Documentation improvements

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Original FP-Growth paper: Han et al., "Mining Frequent Patterns without Candidate Generation" (2000)
- Inspired by mlxtend and efficient-apriori implementations
- Built with PyO3, ndarray, and Rayon

## 📚 References

- [FP-Growth Paper](https://www.cs.sfu.ca/~han/DM_Book_2nd_Edition/SlideChap6.pdf)
- [PyO3 Documentation](https://pyo3.rs/)
- [Benchmarking Guide](BENCHMARKING.md)

---

**Made with ❤️ and Rust** 🦀
