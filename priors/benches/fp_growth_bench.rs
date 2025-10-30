use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use numpy::ndarray::Array2;
use rand::Rng;

// Import from the library
use priors::fp::fp_growth_algorithm;

/// Generate synthetic transaction data
///
/// Parameters:
/// - num_transactions: Number of transactions
/// - num_items: Total number of possible items
/// - avg_transaction_size: Average items per transaction
/// - density: How dense the data is (0.0-1.0)
fn generate_transactions(
    num_transactions: usize,
    num_items: usize,
    avg_transaction_size: usize,
    density: f64,
) -> Array2<i32> {
    let mut rng = rand::thread_rng();
    let mut data = vec![0i32; num_transactions * num_items];

    for tx_idx in 0..num_transactions {
        // Decide how many items in this transaction
        let random_factor: f64 = rng.r#gen();
        let num_items_in_tx = (avg_transaction_size as f64 * (0.5 + random_factor)).round() as usize;
        let num_items_in_tx = num_items_in_tx.min(num_items);

        // Randomly select items (weighted by density)
        for _ in 0..num_items_in_tx {
            let density_check: f64 = rng.r#gen();
            if density_check < density {
                let item = rng.gen_range(0..num_items);
                data[tx_idx * num_items + item] = 1;
            }
        }
    }

    Array2::from_shape_vec((num_transactions, num_items), data).unwrap()
}

/// Benchmark FP-Growth with different dataset sizes
fn bench_fp_growth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_growth_scaling");

    // Different dataset sizes
    let configs = vec![
        ("small_100tx", 100, 20, 5),
        ("medium_500tx", 500, 50, 10),
        ("large_1000tx", 1000, 100, 15),
        ("xlarge_5000tx", 5000, 100, 20),
    ];

    for (name, num_tx, num_items, avg_size) in configs {
        let transactions = generate_transactions(num_tx, num_items, avg_size, 0.7);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &transactions,
            |b, tx| {
                b.iter(|| {
                    fp_growth_algorithm(black_box(tx.view()), black_box(0.1))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FP-Growth with different min_support thresholds
fn bench_fp_growth_min_support(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_growth_min_support");

    let transactions = generate_transactions(1000, 50, 10, 0.7);

    let min_supports = vec![0.05, 0.1, 0.2, 0.3, 0.5];

    for &min_sup in &min_supports {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.2}", min_sup)),
            &min_sup,
            |b, &sup| {
                b.iter(|| {
                    fp_growth_algorithm(black_box(transactions.view()), black_box(sup))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FP-Growth with different data densities
fn bench_fp_growth_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_growth_density");

    let densities = vec![
        ("sparse_30", 0.3),
        ("medium_50", 0.5),
        ("dense_70", 0.7),
        ("very_dense_90", 0.9),
    ];

    for (name, density) in densities {
        let transactions = generate_transactions(1000, 50, 10, density);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &transactions,
            |b, tx| {
                b.iter(|| {
                    fp_growth_algorithm(black_box(tx.view()), black_box(0.1))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark with real-world-like patterns
fn bench_fp_growth_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_growth_patterns");

    // Simulate different shopping patterns

    // 1. Frequent itemsets pattern (grocery shopping)
    let grocery = generate_transactions(1000, 30, 8, 0.8);
    group.bench_with_input(
        BenchmarkId::from_parameter("grocery_pattern"),
        &grocery,
        |b, tx| {
            b.iter(|| fp_growth_algorithm(black_box(tx.view()), black_box(0.15)));
        },
    );

    // 2. Long-tail pattern (e-commerce)
    let ecommerce = generate_transactions(1000, 100, 5, 0.4);
    group.bench_with_input(
        BenchmarkId::from_parameter("ecommerce_longtail"),
        &ecommerce,
        |b, tx| {
            b.iter(|| fp_growth_algorithm(black_box(tx.view()), black_box(0.05)));
        },
    );

    // 3. Uniform pattern (sensor data)
    let sensor = generate_transactions(1000, 20, 15, 0.9);
    group.bench_with_input(
        BenchmarkId::from_parameter("sensor_uniform"),
        &sensor,
        |b, tx| {
            b.iter(|| fp_growth_algorithm(black_box(tx.view()), black_box(0.2)));
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_fp_growth_scaling,
    bench_fp_growth_min_support,
    bench_fp_growth_density,
    bench_fp_growth_patterns
);
criterion_main!(benches);
