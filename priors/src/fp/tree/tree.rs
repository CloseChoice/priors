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

    pub fn new_item(item: usize, parent: Option<usize>) -> Self {
        Self {
            item: Some(item),
            count,
            parent,
            children: HashMap::new(),
        }
    }
}

impl FPTree {
    pub fn new() -> Self {
        let mut nodes = Vec::new();
        let root = FPNode::new_root();
        nodes.push(root);

        Self {
            nodes,
            header_table: HashMap::new(),
            root_index: 0,
        }
    }
}
