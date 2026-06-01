//! Native MCTS engine (Option A): owns a worker's games (boards + trees +
//! per-game history + RNG) and runs PUCT self-play. Python drives only the
//! batched GPU handoff:
//!
//!     while not engine.all_done():
//!         while (b := engine.pending_positions()) is not None:
//!             logits, values = evaluator.evaluate(b)
//!             engine.apply_evals(logits, values)
//!         engine.step_moves()
//!     results = engine.take_results()
//!
//! A pure port of training/mcts.py + selfplay.play_batch (no virtual loss).
//! All search math is f64 so it matches the f64 Python reference bit-for-bit
//! (numpy-f64 exp == libm == Rust f64::exp), enabling exact parity tests.

use crate::{pack_move, Board};
use numpy::ndarray::{Array1, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Dirichlet, Distribution};

// --- constants, mirrored verbatim from mcts.py / selfplay.py ---
const C_PUCT: f64 = 1.5;
const DIRICHLET_ALPHA: f64 = 0.3;
const DIRICHLET_EPS: f64 = 0.25;
const TEMP_OPENING: f64 = 1.0;
const TEMP_MOVES: usize = 20;
const TEMP_FLOOR: f64 = 0.35;
const TEMP_ANNEAL: usize = 40;
const MAX_PLIES: usize = 256;
const POLICY_SIZE: usize = 4672;
const ENC_LEN: usize = 18 * 8 * 8;

struct Node {
    prior: f64,
    visit_count: u32,
    value_sum: f64,
    children: Vec<(u16, u32)>, // (move handle, child arena idx) — insertion order
    expanded: bool,
    noised: bool,
    move_index: i32,
}

impl Node {
    fn new(prior: f64, move_index: i32) -> Self {
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
    fn q(&self) -> f64 {
        if self.visit_count > 0 {
            self.value_sum / self.visit_count as f64
        } else {
            0.0
        }
    }
}

struct Game {
    board: Board,
    arena: Vec<Node>, // root at index 0
    history: Vec<(Vec<f32>, Vec<f32>, bool)>, // (state[1152], pi[4672], turn)
    done: bool,
    result: i8, // white POV: +1 / -1 / 0
}

impl Game {
    fn from_board(board: Board) -> Self {
        Game {
            board,
            arena: vec![Node::new(0.0, -1)],
            history: Vec::new(),
            done: false,
            result: 0,
        }
    }
    fn temperature(&self) -> f64 {
        let ply = self.history.len();
        if ply < TEMP_MOVES {
            return TEMP_OPENING;
        }
        let frac = (((ply - TEMP_MOVES) as f64) / TEMP_ANNEAL as f64).min(1.0);
        TEMP_OPENING + frac * (TEMP_FLOOR - TEMP_OPENING)
    }
}

fn argmax(xs: &[f64]) -> usize {
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

fn select_child(arena: &[Node], node_idx: usize) -> Option<(u16, usize)> {
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
fn walk_to_leaf(arena: &[Node], root_idx: usize, board: &mut Board) -> Vec<usize> {
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
fn expand_node(arena: &mut Vec<Node>, node_idx: usize, board: &Board, logits: &[f32]) {
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

fn backprop(arena: &mut [Node], path: &[usize], value: f64) {
    let mut v = value;
    for &node_idx in path.iter().rev() {
        let n = &mut arena[node_idx];
        n.visit_count += 1;
        n.value_sum += v;
        v = -v;
    }
}

/// Idempotent Dirichlet noise at the root (once per node).
fn add_dirichlet_noise(arena: &mut [Node], root_idx: usize, rng: &mut StdRng) {
    if arena[root_idx].noised || arena[root_idx].children.is_empty() {
        return;
    }
    let child_idxs: Vec<u32> = arena[root_idx].children.iter().map(|&(_, c)| c).collect();
    let k = child_idxs.len();
    let noise: Vec<f64> = if k == 1 {
        vec![1.0]
    } else {
        Dirichlet::new(&vec![DIRICHLET_ALPHA; k])
            .unwrap()
            .sample(rng)
    };
    for (&ci, &n) in child_idxs.iter().zip(noise.iter()) {
        let c = &mut arena[ci as usize];
        c.prior = (1.0 - DIRICHLET_EPS) * c.prior + DIRICHLET_EPS * n;
    }
    arena[root_idx].noised = true;
}

fn visits_to_pi(arena: &[Node], root_idx: usize, temperature: f64) -> Vec<f32> {
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

fn sample_move(arena: &[Node], root_idx: usize, temperature: f64, rng: &mut StdRng) -> u16 {
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
fn copy_subtree(old: &[Node], idx: usize, new: &mut Vec<Node>) -> usize {
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

enum Phase {
    RootExpand,
    Sim(usize),
    SimsDone,
}

struct Leaf {
    game: usize,
    path: Vec<usize>,
    board: Board,
    terminal: Option<f64>,
    eval_row: Option<usize>,
}

enum Pending {
    None,
    Roots(Vec<usize>),
    Leaves(Vec<Leaf>),
}

#[pyclass]
pub struct MctsEngine {
    games: Vec<Game>,
    sims: usize,
    add_root_noise: bool,
    rng: StdRng,
    phase: Phase,
    pending: Pending,
    results: Vec<(Vec<f32>, Vec<f32>, f32)>,
}

impl MctsEngine {
    fn apply_root_noise(&mut self) {
        if !self.add_root_noise {
            return;
        }
        for gi in 0..self.games.len() {
            if !self.games[gi].done && self.games[gi].arena[0].expanded {
                add_dirichlet_noise(&mut self.games[gi].arena, 0, &mut self.rng);
            }
        }
    }

    fn advance_subtree(&mut self, gi: usize, handle: u16) {
        let child = self.games[gi].arena[0]
            .children
            .iter()
            .find(|&&(h, _)| h == handle)
            .map(|&(_, c)| c as usize);
        match child {
            Some(ci) => {
                let mut new_arena = Vec::new();
                copy_subtree(&self.games[gi].arena, ci, &mut new_arena);
                self.games[gi].arena = new_arena;
            }
            None => {
                self.games[gi].arena = vec![Node::new(0.0, -1)];
            }
        }
    }

    fn finalize(&mut self, gi: usize) {
        let z_white = self.games[gi].result as f32;
        let hist = std::mem::take(&mut self.games[gi].history);
        for (state, pi, turn) in hist {
            let z = if turn { z_white } else { -z_white };
            self.results.push((state, pi, z));
        }
    }

    /// Replace every finished game with a fresh start-position game (steady
    /// game-refill: keeps the in-flight batch full instead of shrinking to a
    /// thin tail as games finish). The shared engine rng keeps advancing, so
    /// the stream stays deterministic for a given seed.
    fn refill_done(&mut self) {
        for g in self.games.iter_mut() {
            if g.done {
                *g = Game::from_board(Board::new());
            }
        }
    }
}

#[pymethods]
impl MctsEngine {
    #[new]
    fn new(n_games: usize, sims: usize, seed: u64, add_root_noise: bool) -> Self {
        let games = (0..n_games)
            .map(|_| Game::from_board(Board::new()))
            .collect();
        MctsEngine {
            games,
            sims,
            add_root_noise,
            rng: StdRng::seed_from_u64(seed),
            phase: Phase::RootExpand,
            pending: Pending::None,
            results: Vec::new(),
        }
    }

    #[staticmethod]
    fn from_fens(fens: Vec<String>, sims: usize, seed: u64, add_root_noise: bool) -> PyResult<Self> {
        let mut games = Vec::with_capacity(fens.len());
        for f in &fens {
            games.push(Game::from_board(Board::from_fen(f)?));
        }
        Ok(MctsEngine {
            games,
            sims,
            add_root_noise,
            rng: StdRng::seed_from_u64(seed),
            phase: Phase::RootExpand,
            pending: Pending::None,
            results: Vec::new(),
        })
    }

    /// Advance until a batch needs NN evaluation; return it as (M,18,8,8) f32,
    /// or None when this cycle's sims are done (caller should `step_moves`).
    fn pending_positions<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f32>>> {
        loop {
            match self.phase {
                Phase::RootExpand => {
                    let idxs: Vec<usize> = (0..self.games.len())
                        .filter(|&i| !self.games[i].done && !self.games[i].arena[0].expanded)
                        .collect();
                    if idxs.is_empty() {
                        self.apply_root_noise();
                        self.phase = Phase::Sim(0);
                        continue;
                    }
                    let m = idxs.len();
                    let mut buf = vec![0.0f32; m * ENC_LEN];
                    for (row, &gi) in idxs.iter().enumerate() {
                        self.games[gi]
                            .board
                            .encode_into(&mut buf[row * ENC_LEN..(row + 1) * ENC_LEN]);
                    }
                    self.pending = Pending::Roots(idxs);
                    return Some(
                        Array4::from_shape_vec((m, 18, 8, 8), buf)
                            .unwrap()
                            .into_pyarray(py),
                    );
                }
                Phase::Sim(s) => {
                    if s >= self.sims {
                        self.phase = Phase::SimsDone;
                        continue;
                    }
                    let mut leaves: Vec<Leaf> = Vec::new();
                    let mut buf: Vec<f32> = Vec::new();
                    let mut row = 0usize;
                    for gi in 0..self.games.len() {
                        if self.games[gi].done {
                            continue;
                        }
                        let mut board = self.games[gi].board.clone_search();
                        let path = walk_to_leaf(&self.games[gi].arena, 0, &mut board);
                        let terminal = board.terminal_value().map(|x| x as f64);
                        let eval_row = if terminal.is_none() {
                            let start = buf.len();
                            buf.resize(start + ENC_LEN, 0.0);
                            board.encode_into(&mut buf[start..start + ENC_LEN]);
                            let r = row;
                            row += 1;
                            Some(r)
                        } else {
                            None
                        };
                        leaves.push(Leaf {
                            game: gi,
                            path,
                            board,
                            terminal,
                            eval_row,
                        });
                    }
                    if row == 0 {
                        for lf in &leaves {
                            backprop(&mut self.games[lf.game].arena, &lf.path, lf.terminal.unwrap());
                        }
                        self.phase = Phase::Sim(s + 1);
                        continue;
                    }
                    self.pending = Pending::Leaves(leaves);
                    return Some(
                        Array4::from_shape_vec((row, 18, 8, 8), buf)
                            .unwrap()
                            .into_pyarray(py),
                    );
                }
                Phase::SimsDone => return None,
            }
        }
    }

    fn apply_evals(&mut self, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) {
        let logits = logits.as_array();
        let values = values.as_array();
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::Roots(idxs) => {
                for (row, &gi) in idxs.iter().enumerate() {
                    let lr = logits.row(row);
                    let lrs = lr.as_slice().unwrap();
                    let game = &mut self.games[gi];
                    expand_node(&mut game.arena, 0, &game.board, lrs);
                }
                self.apply_root_noise();
                self.phase = Phase::Sim(0);
            }
            Pending::Leaves(leaves) => {
                for lf in &leaves {
                    let v = match lf.terminal {
                        Some(t) => t,
                        None => {
                            let r = lf.eval_row.unwrap();
                            let lr = logits.row(r);
                            let lrs = lr.as_slice().unwrap();
                            let leaf_node = *lf.path.last().unwrap();
                            let game = &mut self.games[lf.game];
                            expand_node(&mut game.arena, leaf_node, &lf.board, lrs);
                            values[r] as f64
                        }
                    };
                    backprop(&mut self.games[lf.game].arena, &lf.path, v);
                }
                if let Phase::Sim(s) = self.phase {
                    self.phase = Phase::Sim(s + 1);
                }
            }
            Pending::None => {}
        }
    }

    /// One self-play move per active game: record (state, pi, turn), sample +
    /// push, advance the subtree, and finalize finished games. Resets the cycle.
    fn step_moves(&mut self) {
        for gi in 0..self.games.len() {
            if self.games[gi].done {
                continue;
            }
            let temp_pi = self.games[gi].temperature();
            let pi = visits_to_pi(&self.games[gi].arena, 0, temp_pi);
            let mut state = vec![0.0f32; ENC_LEN];
            self.games[gi].board.encode_into(&mut state);
            let turn = self.games[gi].board.turn();
            self.games[gi].history.push((state, pi, turn));

            let temp_s = self.games[gi].temperature();
            let handle = sample_move(&self.games[gi].arena, 0, temp_s, &mut self.rng);
            self.games[gi].board.push(handle);
            self.advance_subtree(gi, handle);

            let res = self.games[gi].board.outcome_white();
            if res.is_some() || self.games[gi].history.len() >= MAX_PLIES {
                self.games[gi].result = res.unwrap_or(0);
                self.games[gi].done = true;
                self.finalize(gi);
            }
        }
        self.phase = Phase::RootExpand;
    }

    fn all_done(&self) -> bool {
        self.games.iter().all(|g| g.done)
    }

    /// Drain finished (state(18,8,8) f32, pi(4672,) f32, z f32) training tuples.
    fn take_results<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Vec<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray1<f32>>, f32)> {
        let drained = std::mem::take(&mut self.results);
        drained
            .into_iter()
            .map(|(state, pi, z)| {
                let s = Array3::from_shape_vec((18, 8, 8), state).unwrap().into_pyarray(py);
                let p = Array1::from_vec(pi).into_pyarray(py);
                (s, p, z)
            })
            .collect()
    }

    /// Drive the entire self-play loop in Rust (option B — in-process forward).
    ///
    /// `forward(batch)` is a Python callable taking an (M,18,8,8) f32 ndarray
    /// and returning (logits[M,4672] f32, values[M] f32). It is the ONLY
    /// crossing back into Python — the in-process GPU forward.
    /// `result_sink(rows, n_finished)` receives a list of finished
    /// (state(18,8,8), pi(4672,), z) tuples plus the number of games that
    /// finished this cycle (push them onto the trainer queue). `stop()` returns
    /// True to end the loop (orphan/parent-death + shutdown checks live there).
    ///
    /// refill=True: run until `stop`, refilling finished games so the batch stays
    /// full. refill=False: stop when every game is done (the equivalence test).
    fn run(
        &mut self,
        py: Python<'_>,
        forward: PyObject,
        result_sink: PyObject,
        stop: PyObject,
        refill: bool,
    ) -> PyResult<()> {
        loop {
            // Inner micro-loop: feed every NN batch this cycle needs through the
            // in-process forward. This is the ~sims-per-ply hot path (the ~23k
            // round-trips/game that used to cross the process boundary).
            loop {
                let batch = match self.pending_positions(py) {
                    Some(b) => b,
                    None => break,
                };
                let out = forward.call1(py, (batch,))?;
                let bound = out.bind(py);
                let (logits, values): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>) =
                    bound.extract()?;
                self.apply_evals(logits, values);
            }
            self.step_moves();

            // Every game is refilled each cycle, so any game still `done` here
            // finished THIS cycle — that's the finished-game count for stats.
            if !self.results.is_empty() {
                let n_finished = self.games.iter().filter(|g| g.done).count();
                let rows = self.take_results(py);
                let list = PyList::new(py, rows)?;
                result_sink.call1(py, (list, n_finished))?;
            }

            if refill {
                self.refill_done();
            } else if self.all_done() {
                break;
            }
            if stop.call0(py)?.bind(py).extract::<bool>()? {
                break;
            }
        }
        Ok(())
    }

    // ---- parity / eval accessors ----

    /// Per root-child (move_index, visit_count, value_sum) for a game.
    fn root_child_stats(&self, gi: usize) -> Vec<(i32, u32, f64)> {
        let arena = &self.games[gi].arena;
        arena[0]
            .children
            .iter()
            .map(|&(_, c)| {
                let n = &arena[c as usize];
                (n.move_index, n.visit_count, n.value_sum)
            })
            .collect()
    }

    fn root_pi<'py>(&self, py: Python<'py>, gi: usize, temperature: f64) -> Bound<'py, PyArray1<f32>> {
        Array1::from_vec(visits_to_pi(&self.games[gi].arena, 0, temperature)).into_pyarray(py)
    }

    /// Temperature-0 selected move handle per game (for eval-mode / parity).
    fn select_moves(&mut self, temperature: f64) -> Vec<u16> {
        (0..self.games.len())
            .map(|gi| sample_move(&self.games[gi].arena, 0, temperature, &mut self.rng))
            .collect()
    }

    /// Re-arm a fresh simulation cycle (used after advancing externally).
    fn reset_cycle(&mut self) {
        self.phase = Phase::RootExpand;
    }

    /// Test/eval hook: advance a game by the root child with the given policy
    /// index (push its move + reuse the subtree). Returns false if no such child.
    fn advance_by_move_index(&mut self, gi: usize, move_index: i32) -> bool {
        let handle = {
            let arena = &self.games[gi].arena;
            arena[0]
                .children
                .iter()
                .find(|&&(_, c)| arena[c as usize].move_index == move_index)
                .map(|&(h, _)| h)
        };
        match handle {
            Some(h) => {
                self.games[gi].board.push(h);
                self.advance_subtree(gi, h);
                true
            }
            None => false,
        }
    }
}
