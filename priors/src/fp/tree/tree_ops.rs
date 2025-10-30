use super::tree::{FPNode, FPTree};

impl FPTree {
    pub fn insert_transaction(&mut self, transaction: &[usize], counts: &[usize]) {
        let mut current_index = self.root_index;

        for (&item, &count) in transaction.iter().zip(counts.iter()) {
            let current_node = &self.nodes[current_index];

            if let Some(&child_index) = current_node.children.get(&item) {
                self.nodes[child_index].count += count;
                current_index = child_index;
            } else {
                let new_node = FPNode::new_item(item, Some(current_index));
                let new_index = self.nodes.len();
                self.nodes.push(new_node);

                self.nodes[current_index].children.insert(item, new_index);

                self.header_table
                    .entry(item)
                    .or_insert_with(Vec::new)
                    .push(new_index);

                self.nodes[new_index].count += count;
                current_index = new_index;
            }
        }
    }

    pub fn get_prefix_paths(&self, item: usize) -> Vec<(Vec<usize>, usize)> {
        let mut paths = Vec::new();

        if let Some(node_indices) = self.header_table.get(&item) {
            for &node_index in node_indices {
                let node = &self.nodes[node_index];
                let count = node.count;

                let mut path = Vec::new();
                let mut current_index = node.parent;

                while let Some(idx) = current_index {
                    let parent_node = &self.nodes[parent_index];
                    if let Some(parent_item) = parent_node.item {
                        path.push(parent_item);
                    }
                    current_index = parent_node.parent;
                }
                path.reverse();
                if !path.is_empty() {
                    paths.push((path, count));
                }
            }
        }

        paths
    }

    pub fn has_single_path(&self) -> bool {
        let mut current_index = self.root_index;

        loop {
            let current_index = &self.nodes[current_index];

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
