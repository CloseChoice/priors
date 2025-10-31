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

    pub fn allocate(&self, bytes: usize) -> Result<(), MemoryError> {
        let current = self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
        if current + bytes > self.max_bytes {
            self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
            Err(MemoryError::BudgetExceeded)
        } else {
            Ok(())
        }
    }

    pub fn deallocate(&self, bytes: usize) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub enum MemoryError {
    BudgetExceeded,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Memory budget exceeded")
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

impl Drop for MemoryGuard<'_> {
    fn drop(&mut self) {
        self.budget.deallocate(self.bytes);
    }
}
