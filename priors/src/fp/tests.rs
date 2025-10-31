use super::*;
use numpy::ndarray::Array2;

#[test]
fn test_itemset_storage() {
    let mut storage = storage::ItemsetStorage::new();
    storage.add_itemset(vec![7, 2, 5]);
    storage.add_itemset(vec![1, 3]);
    storage.add_itemset(vec![2, 3, 5, 9]);

    assert_eq!(storage.get_itemset(0), &[2, 5, 7]);
    assert_eq!(storage.get_itemset(1), &[1, 3]);
    assert_eq!(storage.get_itemset(2), &[2, 3, 5, 9]);
    assert_eq!(storage.len(), 3);
}

#[test]
fn test_frequent_level() {
    let mut level = storage::FrequentLevel::new(2);
    level.add_itemset(vec![1, 2]);
    level.add_itemset(vec![3, 4]);

    assert_eq!(level.len(), 2);
    assert_eq!(level.itemset_size, 2);
    assert_eq!(level.iter_itemsets().count(), 2);
}

#[test]
fn test_fp_tree() {
    let mut tree = tree::FPTree::new();
    tree.insert_transaction(&[1, 2, 3], &[1, 1, 1]);

    assert!(tree.nodes[0].children.contains_key(&1));
    assert_eq!(tree.header_table.get(&1).unwrap().len(), 1);

    tree.insert_transaction(&[1, 2, 4], &[1, 1, 1]);
    let node1_idx = tree.nodes[0].children[&1];
    assert_eq!(tree.nodes[node1_idx].count, 2);
}

#[test]
fn test_fp_tree_prefix_paths() {
    let mut tree = tree::FPTree::new();
    tree.insert_transaction(&[1, 2, 3], &[1, 1, 1]);
    tree.insert_transaction(&[1, 2, 4], &[1, 1, 1]);

    let paths = tree.get_prefix_paths(3);
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].0, vec![1, 2]);
    assert_eq!(paths[0].1, 1);
}

#[test]
fn test_fp_tree_single_path() {
    let mut tree1 = tree::FPTree::new();
    tree1.insert_transaction(&[1, 2, 3], &[1, 1, 1]);
    assert!(tree1.has_single_path());

    let path = tree1.get_single_path();
    assert_eq!(path.len(), 3);

    let mut tree2 = tree::FPTree::new();
    tree2.insert_transaction(&[1, 2], &[1, 1]);
    tree2.insert_transaction(&[1, 3], &[1, 1]);
    assert!(!tree2.has_single_path());
}

#[test]
fn test_fp_growth() {
    let transactions = Array2::from_shape_vec(
        (4, 3),
        vec![
            1, 1, 0,
            1, 1, 1,
            1, 0, 1,
            0, 1, 1,
        ],
    ).unwrap();

    let result = mining::fp_growth_algorithm(transactions.view(), 0.5);
    assert!(!result.is_empty());
    let total: usize = result.iter().map(|l| l.len()).sum();
    assert!(total > 0);
}

#[test]
fn test_conditional_tree() {
    let prefix_paths = vec![
        (vec![1, 2], 2),
        (vec![1], 1),
    ];

    let cond_tree = builder::build_conditional_fp_tree(&prefix_paths, 2);
    assert!(cond_tree.header_table.contains_key(&1));
}
