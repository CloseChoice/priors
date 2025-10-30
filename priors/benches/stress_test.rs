use priors::fp::{fp_growth_algorithm, LazyFPGrowth};
use ndarray::Array2;
use rand::Rng;
use std::time::Instant;

fn generate_transactions(
    num_transactions: usize,
    num_items: usize,
    avg_transaction_size: usize,
    density: f64,
) -> Array2<i32> {
    let mut rng = rand::thread_rng();
    let mut data = vec![0i32; num_transactions * num_items];

    for tx_idx in 0..num_transactions {
        let random_factor: f64 = rng.r#gen();
        let num_items_in_tx = (avg_transaction_size as f64 * (0.5 + random_factor)).round() as usize;
        let num_items_in_tx = num_items_in_tx.min(num_items);

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

fn print_memory_stats() {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(rss) = String::from_utf8(output.stdout) {
                if let Ok(kb) = rss.trim().parse::<usize>() {
                    println!("  Memory: {} MB", kb / 1024);
                }
            }
        }
    }
}

fn stress_test_oom_scenario() {
    println!("\n=== OOM Stress Test ===");

    let configs = vec![
        ("10K x 50", 10_000, 50, 15),
        ("50K x 80", 50_000, 80, 20),
        ("100K x 100", 100_000, 100, 25),
        ("200K x 150", 200_000, 150, 30),
    ];

    for (name, num_tx, num_items, avg_size) in configs {
        println!("\nTesting: {}", name);
        println!("  Generating {} transactions...", num_tx);

        let start_gen = Instant::now();
        let transactions = generate_transactions(num_tx, num_items, avg_size, 0.7);
        println!("  Generated in {:?}", start_gen.elapsed());
        print_memory_stats();

        println!("  Running FP-Growth (min_support=0.01)...");
        let start = Instant::now();

        match std::panic::catch_unwind(|| {
            fp_growth_algorithm(transactions.view(), 0.01)
        }) {
            Ok(result) => {
                let elapsed = start.elapsed();
                let total_patterns: usize = result.iter().map(|l| l.len()).sum();
                println!("  ✓ Completed in {:?}", elapsed);
                println!("  Found {} patterns", total_patterns);
                print_memory_stats();
            }
            Err(_) => {
                println!("  ✗ OOM or panic occurred!");
            }
        }
    }
}

fn stress_test_lazy_vs_regular() {
    println!("\n=== Lazy vs Regular Comparison ===");

    let configs = vec![
        ("50K x 100", 50_000, 100, 20),
        ("100K x 150", 100_000, 150, 25),
    ];

    for (name, num_tx, num_items, avg_size) in configs {
        println!("\nDataset: {}", name);
        let transactions = generate_transactions(num_tx, num_items, avg_size, 0.7);
        let min_support = 0.01;

        println!("  Regular FP-Growth:");
        let start = Instant::now();
        let regular_result = fp_growth_algorithm(transactions.view(), min_support);
        let regular_time = start.elapsed();
        let regular_patterns: usize = regular_result.iter().map(|l| l.len()).sum();
        println!("    Time: {:?}", regular_time);
        println!("    Patterns: {}", regular_patterns);
        print_memory_stats();

        println!("  Lazy FP-Growth (chunked):");
        let start = Instant::now();
        let mut lazy = LazyFPGrowth::new();

        let chunk_size = 5000;
        let num_chunks = (num_tx + chunk_size - 1) / chunk_size;

        for chunk_idx in 0..num_chunks {
            let start_idx = chunk_idx * chunk_size;
            let end_idx = ((chunk_idx + 1) * chunk_size).min(num_tx);
            let chunk = transactions.slice(ndarray::s![start_idx..end_idx, ..]);
            lazy.count_pass(chunk);
        }

        lazy.finalize_counts(min_support);

        for chunk_idx in 0..num_chunks {
            let start_idx = chunk_idx * chunk_size;
            let end_idx = ((chunk_idx + 1) * chunk_size).min(num_tx);
            let chunk = transactions.slice(ndarray::s![start_idx..end_idx, ..]);
            lazy.build_pass(chunk);
        }

        let lazy_result = lazy.mine_patterns(min_support).unwrap();
        let lazy_time = start.elapsed();
        let lazy_patterns: usize = lazy_result.iter().map(|l| l.len()).sum();

        println!("    Time: {:?}", lazy_time);
        println!("    Patterns: {}", lazy_patterns);
        print_memory_stats();

        let overhead = (lazy_time.as_secs_f64() / regular_time.as_secs_f64() - 1.0) * 100.0;
        println!("    Overhead: {:.1}%", overhead);
    }
}

fn stress_test_extreme_low_support() {
    println!("\n=== Extreme Low Support Test ===");

    let transactions = generate_transactions(20_000, 100, 20, 0.6);

    let support_levels = vec![0.05, 0.02, 0.01, 0.005, 0.001];

    for &min_support in &support_levels {
        println!("\nTesting min_support = {}", min_support);
        let start = Instant::now();

        match std::panic::catch_unwind(|| {
            fp_growth_algorithm(transactions.view(), min_support)
        }) {
            Ok(result) => {
                let elapsed = start.elapsed();
                let total_patterns: usize = result.iter().map(|l| l.len()).sum();
                let max_level = result.len();
                println!("  Time: {:?}", elapsed);
                println!("  Patterns: {}", total_patterns);
                println!("  Max itemset size: {}", max_level);
                print_memory_stats();

                if total_patterns > 1_000_000 {
                    println!("  ⚠ Pattern explosion detected!");
                }
            }
            Err(_) => {
                println!("  ✗ Failed (likely OOM)");
            }
        }
    }
}

fn stress_test_dense_data() {
    println!("\n=== Dense Data Test (worst case) ===");

    let configs = vec![
        ("Dense 80%", 10_000, 50, 40, 0.8),
        ("Dense 90%", 10_000, 50, 45, 0.9),
        ("Dense 95%", 10_000, 50, 47, 0.95),
    ];

    for (name, num_tx, num_items, avg_size, density) in configs {
        println!("\nTesting: {}", name);
        let transactions = generate_transactions(num_tx, num_items, avg_size, density);

        let start = Instant::now();
        match std::panic::catch_unwind(|| {
            fp_growth_algorithm(transactions.view(), 0.1)
        }) {
            Ok(result) => {
                let elapsed = start.elapsed();
                let total_patterns: usize = result.iter().map(|l| l.len()).sum();
                println!("  Time: {:?}", elapsed);
                println!("  Patterns: {}", total_patterns);
                print_memory_stats();
            }
            Err(_) => {
                println!("  ✗ Failed");
            }
        }
    }
}

fn main() {
    println!("=== FP-Growth Stress Testing Suite ===");
    println!("Testing memory limits, OOM scenarios, and performance degradation\n");

    stress_test_oom_scenario();
    stress_test_lazy_vs_regular();
    stress_test_extreme_low_support();
    stress_test_dense_data();

    println!("\n=== Stress Testing Complete ===");
}
