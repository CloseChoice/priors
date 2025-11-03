#[derive(Debug, Clone)]
pub struct ItemsetStorage {
    pub items: Vec<usize>,
    pub offsets: Vec<(usize, usize)>,
    pub supports: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FrequentLevel {
    pub storage: ItemsetStorage,
    pub itemset_size: usize,
}

impl ItemsetStorage {
    pub(crate) fn new() -> Self {
        Self { items: Vec::new(), offsets: Vec::new(), supports: Vec::new() }
    }

    pub(crate) fn add_itemset(&mut self, items: Vec<usize>) {
        self.add_itemset_with_support(items, 0);
    }

    pub(crate) fn add_itemset_with_support(&mut self, mut items: Vec<usize>, support: usize) {
        items.sort_unstable();
        items.dedup();
        let start = self.items.len();
        self.items.extend_from_slice(&items);
        self.offsets.push((start, items.len()));
        self.supports.push(support);
    }

    pub(crate) fn get_itemset(&self, idx: usize) -> &[usize] {
        let (start, len) = self.offsets[idx];
        &self.items[start..start + len]
    }

    pub(crate) fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl FrequentLevel {
    pub fn new(itemset_size: usize) -> Self {
        Self { storage: ItemsetStorage::new(), itemset_size }
    }

    pub fn add_itemset(&mut self, items: Vec<usize>) -> usize {
        self.storage.add_itemset(items);
        self.storage.len() - 1
    }

    pub fn add_itemset_with_support(&mut self, items: Vec<usize>, support: usize) -> usize {
        self.storage.add_itemset_with_support(items, support);
        self.storage.len() - 1
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn get_itemset(&self, idx: usize) -> &[usize] {
        self.storage.get_itemset(idx)
    }

    pub fn iter_itemsets(&self) -> impl Iterator<Item = &[usize]> {
        (0..self.storage.len()).map(move |idx| self.get_itemset(idx))
    }
}
