use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MemoryBudget {
    max_bytes: usize,
    current_bytes: AtomicUsize,
}

impl MemoryBudget {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: AtomicUsize::new(0),
        }
    }

    pub fn unlimited() -> Self {
        Self::new(usize::MAX)
    }

    pub fn allocate(&self, bytes: usize) -> Result<(), MemoryError> {
        let current = self.current_bytes.fetch_add(bytes, Ordering::SeqCst);
        if current + bytes > self.max_bytes {
            self.current_bytes.fetch_sub(bytes, Ordering::SeqCst);
            return Err(MemoryError::BudgetExceeded {
                requested: bytes,
                available: self.max_bytes.saturating_sub(current),
            });
        }
        Ok(())
    }

    pub fn deallocate(&self, bytes: usize) {
        self.current_bytes.fetch_sub(bytes, Ordering::SeqCst);
    }

    pub fn current_usage(&self) -> usize {
        self.current_bytes.load(Ordering::SeqCst)
    }

    pub fn available(&self) -> usize {
        self.max_bytes.saturating_sub(self.current_usage())
    }

    pub fn usage_percentage(&self) -> f64 {
        (self.current_usage() as f64 / self.max_bytes as f64) * 100.0
    }
}

#[derive(Debug, Clone)]
pub enum MemoryError {
    BudgetExceeded { requested: usize, available: usize },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::BudgetExceeded { requested, available } => {
                write!(
                    f,
                    "Memory budget exceeded: requested {} bytes, {} available",
                    requested, available
                )
            }
        }
    }
}

impl std::error::Error for MemoryError {}

pub struct MemoryGuard<'a> {
    budget: &'a MemoryBudget,
    bytes: usize,
}

impl<'a> MemoryGuard<'a> {
    pub fn new(budget: &'a MemoryBudget, bytes: usize) -> Result<Self, MemoryError> {
        budget.allocate(bytes)?;
        Ok(Self { budget, bytes })
    }
}

impl<'a> Drop for MemoryGuard<'a> {
    fn drop(&mut self) {
        self.budget.deallocate(self.bytes);
    }
}

pub fn estimate_fp_tree_size(num_nodes: usize, avg_items_per_node: usize) -> usize {
    let node_size = std::mem::size_of::<usize>() * 3 + std::mem::size_of::<Vec<usize>>();
    let header_overhead = 64;
    num_nodes * node_size + avg_items_per_node * std::mem::size_of::<usize>() + header_overhead
}

pub fn estimate_itemset_storage_size(num_itemsets: usize, avg_itemset_size: usize) -> usize {
    let offset_size = std::mem::size_of::<(usize, usize)>();
    let item_size = std::mem::size_of::<usize>();
    num_itemsets * offset_size + num_itemsets * avg_itemset_size * item_size
}
