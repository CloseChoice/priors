use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FPNode {
    pub item: Option<usize>,
    pub count: usize,
    pub parent: Option<usize>,
    pub children: HashMap<usize, usize>,
}

#[derive(Debug, Clone)]
pub struct FPTree {
    pub nodes: Vec<FPNode>,
    pub header_table: HashMap<usize, Vec<usize>>,
    pub root_index: usize,
}

impl FPNode {
    pub fn new_root() -> Self {
        Self {
            item: None,
            count: 0,
            parent: None,
            children: HashMap::new(),
        }
    }

    pub fn new_item(item: usize, count: usize, parent: Option<usize>) -> Self {
        Self {
            item: Some(item),
            count,
            parent,
            children: HashMap::new(),
        }
    }
}

impl Default for FPTree {
    fn default() -> Self {
        Self::new()
    }
}

impl FPTree {
    pub fn new() -> Self {
        let mut nodes = Vec::new();
        nodes.push(FPNode::new_root());
        Self {
            nodes,
            header_table: HashMap::new(),
            root_index: 0,
        }
    }

    pub fn insert_transaction(&mut self, transaction: &[usize], counts: &[usize]) {
        let mut current_index = self.root_index;

        for (&item, &count) in transaction.iter().zip(counts.iter()) {
            if let Some(&child_index) = self.nodes[current_index].children.get(&item) {
                self.nodes[child_index].count += count;
                current_index = child_index;
            } else {
                let new_index = self.nodes.len();
                self.nodes
                    .push(FPNode::new_item(item, count, Some(current_index)));
                self.nodes[current_index].children.insert(item, new_index);
                self.header_table.entry(item).or_default().push(new_index);
                current_index = new_index;
            }
        }
    }

    pub fn get_prefix_paths(&self, item: usize) -> Vec<(Vec<usize>, usize)> {
        self.header_table.get(&item).map_or(Vec::new(), |nodes| {
            nodes
                .iter()
                .filter_map(|&idx| {
                    let mut path = Vec::new();
                    let mut current = self.nodes[idx].parent;

                    while let Some(i) = current {
                        if let Some(item) = self.nodes[i].item {
                            path.push(item);
                        }
                        current = self.nodes[i].parent;
                    }

                    path.reverse();
                    (!path.is_empty()).then_some((path, self.nodes[idx].count))
                })
                .collect()
        })
    }

    pub fn has_single_path(&self) -> bool {
        let mut current_index = self.root_index;

        loop {
            let current_node = &self.nodes[current_index];

            if current_node.children.len() > 1 {
                return false;
            }

            if current_node.children.is_empty() {
                return true;
            }

            current_index = *current_node.children.values().next().unwrap();
        }
    }

    pub fn get_single_path(&self) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        let mut current_index = self.root_index;

        loop {
            let current_node = &self.nodes[current_index];

            if current_node.children.is_empty() {
                break;
            }

            current_index = *current_node.children.values().next().unwrap();
            let child_node = &self.nodes[current_index];

            if let Some(item) = child_node.item {
                path.push((item, child_node.count));
            }
        }
        path
    }
}
