use super::storage::FrequentLevel;

pub fn generate_combinations_from_path(
    path: &[(usize, usize)],
    k: usize,
    alpha: &[usize],
    result: &mut Vec<FrequentLevel>,
) {
    if k == 0 || k > path.len() {
        return;
    }

    let indices: Vec<usize> = (0..path.len()).collect();
    let mut callback = |combination: &[usize]| {
        let mut pattern = alpha.to_vec();
        for &idx in combination {
            pattern.push(path[idx].0);
        }
        add_pattern_to_result(&pattern, result);
    };
    generate_combinations_recursive(&indices, k, 0, &mut Vec::new(), &mut callback);
}

pub fn generate_combinations_recursive<F>(
    items: &[usize],
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    callback: &mut F,
) where
    F: FnMut(&[usize]),
{
    if current.len() == k {
        callback(current);
        return;
    }

    for i in start..items.len() {
        current.push(items[i]);
        generate_combinations_recursive(items, k, i + 1, current, callback);
        current.pop();
    }
}

pub fn add_pattern_to_result(pattern: &[usize], result: &mut Vec<FrequentLevel>) {
    let pattern_size = pattern.len();

    while result.len() < pattern_size {
        result.push(FrequentLevel::new(result.len() + 1));
    }

    if pattern_size > 0 {
        result[pattern_size - 1].add_itemset(pattern.to_vec());
    }
}
