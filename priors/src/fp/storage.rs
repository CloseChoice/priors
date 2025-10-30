/// Memory-efficient itemset storage using flat arrays
#[derive(Debug, Clone)]
pub struct ItemsetStorage {
    items: Vec<usize>,
    offsets: Vec<(usize, usize)>,
}

/// Memory-efficient level storage for frequent itemsets
#[derive(Debug, Clone)]
pub struct FrequentLevel {
    pub(crate) storage: ItemsetStorage,
    pub itemset_size: usize,
}

impl ItemsetStorage {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            offsets: Vec::new(),
        }
    }

    pub fn with_capacity(estimated_items: usize, estimated_itemsets: usize) -> Self {
        Self {
            items: Vec::with_capacity(estimated_items),
            offsets: Vec::with_capacity(estimated_itemsets),
        }
    }

    pub fn add_itemset(&mut self, mut items: Vec<usize>) -> usize {
        items.sort_unstable();
        items.dedup();

        let start_idx = self.items.len();
        let length = items.len();

        self.items.extend_from_slice(&items);
        self.offsets.push((start_idx, length));

        self.offsets.len() - 1
    }

    pub fn get_itemset(&self, idx: usize) -> &[usize] {
        let (start, length) = self.offsets[idx];
        &self.items[start..start + length]
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    #[allow(dead_code)]
    pub fn itemset_len(&self, idx: usize) -> usize {
        self.offsets[idx].1
    }
}

impl FrequentLevel {
    pub fn new(itemset_size: usize) -> Self {
        Self {
            storage: ItemsetStorage::new(),
            itemset_size,
        }
    }

    pub fn with_capacity(itemset_size: usize, estimated_itemsets: usize) -> Self {
        let estimated_items = estimated_items * itemset_size;
        Self {
            storage: ItemsetStorage::with_capacity(estimated_items, estimated_itemsets),
            itemset_size,
        }
    }

    pub fn add_itemset(&mut self, items: Vec<usize>) -> usize {
        debug_assert_eq!(items.len(), self.itemset_size);
        self.storage.add_itemset(items)
    }

    pub fn get_itemset(&self, idx: usize) -> &[usize] {
        self.storage.get_itemset(idx)
    }

    pub fn iter_itemsets(&self) -> impl Iterator<Item = &[usize]> {
        (0..self.storage.len()).map(move |idx| self.get_itemset(idx))
    }
}
