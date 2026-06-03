//! MCTS search primitives (pure; no pyo3). A port of training/mcts.py +
//! selfplay.play_batch (no virtual loss). All search math is f64 so it matches
//! the f64 Python reference bit-for-bit (numpy-f64 exp == libm == Rust f64::exp),
//! enabling exact parity tests.
//!
//! Extracted verbatim from the former `chess_rs/src/mcts.rs`; the pyo3
//! `MctsEngine` that drove these now lives in the `chess_rs` cdylib.

use crate::board::{pack_move, Board};
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

// --- constants, mirrored verbatim from mcts.py / selfplay.py ---
pub const C_PUCT: f64 = 1.5;
pub const DIRICHLET_ALPHA: f64 = 0.3;
pub const DIRICHLET_EPS: f64 = 0.25;
pub const TEMP_OPENING: f64 = 1.0;
pub const TEMP_MOVES: usize = 20;
pub const TEMP_FLOOR: f64 = 0.35;
pub const TEMP_ANNEAL: usize = 40;
pub const MAX_PLIES: usize = 256;
pub const POLICY_SIZE: usize = 4672;
pub const ENC_LEN: usize = 18 * 8 * 8;

pub struct Node {
    pub prior: f64,
    pub visit_count: u32,
    pub value_sum: f64,
    pub children: Vec<(u16, u32)>, // (move handle, child arena idx) — insertion order
    pub expanded: bool,
    pub noised: bool,
    pub move_index: i32,
}

impl Node {
    pub fn new(prior: f64, move_index: i32) -> Self {
        Node {
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: Vec::new(),
            expanded: false,
            noised: false,
            move_index,
        }
    }
    pub fn q(&self) -> f64 {
        if self.visit_count > 0 {
            self.value_sum / self.visit_count as f64
        } else {
            0.0
        }
    }
}

pub struct Game {
    pub board: Board,
    pub arena: Vec<Node>, // root at index 0
    pub history: Vec<(Vec<f32>, Vec<f32>, bool)>, // (state[1152], pi[4672], turn)
    pub done: bool,
    pub result: i8, // white POV: +1 / -1 / 0
}

impl Game {
    pub fn from_board(board: Board) -> Self {
        Game {
            board,
            arena: vec![Node::new(0.0, -1)],
            history: Vec::new(),
            done: false,
            result: 0,
        }
    }
    pub fn temperature(&self) -> f64 {
        let ply = self.history.len();
        if ply < TEMP_MOVES {
            return TEMP_OPENING;
        }
        let frac = (((ply - TEMP_MOVES) as f64) / TEMP_ANNEAL as f64).min(1.0);
        TEMP_OPENING + frac * (TEMP_FLOOR - TEMP_OPENING)
    }
}

pub fn argmax(xs: &[f64]) -> usize {
    let mut bi = 0usize;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

pub fn select_child(arena: &[Node], node_idx: usize) -> Option<(u16, usize)> {
    let node = &arena[node_idx];
    let sqrt_parent = (node.visit_count.max(1) as f64).sqrt();
    let mut best: Option<(u16, usize)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for &(handle, child_idx) in &node.children {
        let child = &arena[child_idx as usize];
        let score =
            -child.q() + C_PUCT * child.prior * sqrt_parent / (1.0 + child.visit_count as f64);
        if score > best_score {
            best_score = score;
            best = Some((handle, child_idx as usize));
        }
    }
    best
}

/// Walk PUCT-greedily on a (cloned, repetition-blind) board until an unexpanded
/// node or a terminal. Returns the path of arena indices.
pub fn walk_to_leaf(arena: &[Node], root_idx: usize, board: &mut Board) -> Vec<usize> {
    let mut path = vec![root_idx];
    let mut node_idx = root_idx;
    while arena[node_idx].expanded {
        if arena[node_idx].children.is_empty() {
            break; // terminal (no legal moves)
        }
        let (handle, child_idx) = select_child(arena, node_idx).unwrap();
        board.push(handle);
        path.push(child_idx);
        node_idx = child_idx;
    }
    path
}

/// f64 softmax over the legal moves of `board`, attaching child priors keyed by
/// move handle (insertion order = legal_moves order, matching the Python dict).
pub fn expand_node(arena: &mut Vec<Node>, node_idx: usize, board: &Board, logits: &[f32]) {
    let moves = board.legal_moves_vec();
    if !moves.is_empty() {
        let idxs: Vec<i32> = moves.iter().map(|&m| board.move_to_index(m)).collect();
        let mut leg: Vec<f64> = idxs.iter().map(|&i| logits[i as usize] as f64).collect();
        let maxv = leg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for x in leg.iter_mut() {
            *x = (*x - maxv).exp();
        }
        let total: f64 = leg.iter().sum();
        for ((m, &idx), &e) in moves.iter().zip(idxs.iter()).zip(leg.iter()) {
            let prior = if total > 0.0 { e / total } else { e };
            let cidx = arena.len() as u32;
            arena.push(Node::new(prior, idx));
            arena[node_idx].children.push((pack_move(*m), cidx));
        }
    }
    arena[node_idx].expanded = true;
}

pub fn backprop(arena: &mut [Node], path: &[usize], value: f64) {
    let mut v = value;
    for &node_idx in path.iter().rev() {
        let n = &mut arena[node_idx];
        n.visit_count += 1;
        n.value_sum += v;
        v = -v;
    }
}

/// Idempotent Dirichlet noise at the root (once per node).
pub fn add_dirichlet_noise(arena: &mut [Node], root_idx: usize, rng: &mut StdRng) {
    if arena[root_idx].noised || arena[root_idx].children.is_empty() {
        return;
    }
    let child_idxs: Vec<u32> = arena[root_idx].children.iter().map(|&(_, c)| c).collect();
    let k = child_idxs.len();
    let noise: Vec<f64> = if k == 1 {
        vec![1.0]
    } else {
        Dirichlet::new(&vec![DIRICHLET_ALPHA; k]).unwrap().sample(rng)
    };
    for (&ci, &n) in child_idxs.iter().zip(noise.iter()) {
        let c = &mut arena[ci as usize];
        c.prior = (1.0 - DIRICHLET_EPS) * c.prior + DIRICHLET_EPS * n;
    }
    arena[root_idx].noised = true;
}

pub fn visits_to_pi(arena: &[Node], root_idx: usize, temperature: f64) -> Vec<f32> {
    let mut pi = vec![0.0f32; POLICY_SIZE];
    let children = &arena[root_idx].children;
    if children.is_empty() {
        return pi;
    }
    let counts: Vec<f64> = children
        .iter()
        .map(|&(_, c)| arena[c as usize].visit_count as f64)
        .collect();
    let idxs: Vec<usize> = children
        .iter()
        .map(|&(_, c)| arena[c as usize].move_index as usize)
        .collect();
    if temperature == 0.0 {
        pi[idxs[argmax(&counts)]] = 1.0;
        return pi;
    }
    let powered: Vec<f64> = counts.iter().map(|&c| c.powf(1.0 / temperature)).collect();
    let total: f64 = powered.iter().sum();
    if total <= 0.0 {
        return pi;
    }
    for (&idx, &p) in idxs.iter().zip(powered.iter()) {
        pi[idx] = (p / total) as f32;
    }
    pi
}

pub fn sample_move(arena: &[Node], root_idx: usize, temperature: f64, rng: &mut StdRng) -> u16 {
    let children = &arena[root_idx].children;
    let counts: Vec<f64> = children
        .iter()
        .map(|&(_, c)| arena[c as usize].visit_count as f64)
        .collect();
    if temperature == 0.0 {
        return children[argmax(&counts)].0;
    }
    let powered: Vec<f64> = counts.iter().map(|&c| c.powf(1.0 / temperature)).collect();
    let total: f64 = powered.iter().sum();
    let r = rng.gen::<f64>() * total;
    let mut acc = 0.0;
    for (i, &p) in powered.iter().enumerate() {
        acc += p;
        if r < acc {
            return children[i].0;
        }
    }
    children[children.len() - 1].0
}

/// Copy the subtree rooted at `root` into a fresh arena (preserving children
/// order), so subtree reuse stays bounded — mirrors `g.root = g.root.children[m]`.
pub fn copy_subtree(old: &[Node], idx: usize, new: &mut Vec<Node>) -> usize {
    let onode = &old[idx];
    let my_idx = new.len();
    new.push(Node {
        prior: onode.prior,
        visit_count: onode.visit_count,
        value_sum: onode.value_sum,
        children: Vec::new(),
        expanded: onode.expanded,
        noised: onode.noised,
        move_index: onode.move_index,
    });
    let edges: Vec<(u16, u32)> = old[idx].children.clone();
    let mut new_children = Vec::with_capacity(edges.len());
    for (h, c) in edges {
        let nc = copy_subtree(old, c as usize, new);
        new_children.push((h, nc as u32));
    }
    new[my_idx].children = new_children;
    my_idx
}
