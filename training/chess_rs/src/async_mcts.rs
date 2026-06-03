//! Multi-threaded async self-play engine (CHESS_BACKEND=rust_async).
//!
//! Goal: use many CPU cores for MCTS while feeding ONE central consumer that
//! does big batched GPU forwards, so CPU tree-walking overlaps GPU inference.
//!
//! Topology:
//!   - `n_games` independent game SLOTS, each with its OWN rng (seeded purely
//!     from master_seed + global_slot_index + generation).
//!   - `n_threads` WORKER threads, each owning a disjoint contiguous shard of
//!     slots. A worker runs the same per-cycle batched MCTS as `MctsEngine`
//!     (root-expand -> sims -> step-move) over its shard, but ships each
//!     sub-batch over an MPSC channel to the consumer instead of running the
//!     forward itself. Worker game state is single-owner (no locks).
//!   - ONE CONSUMER thread (the only thread that touches Python): drains the
//!     channel "wait-for-first-then-drain-to-cap", concatenates sub-batches into
//!     one big batch, runs the Python `forward` under the GIL, and routes the
//!     result rows back to each worker's response channel. Finished games also
//!     flow to the consumer, which calls `result_sink`.
//!
//! Overlap is emergent: in steady state the GPU forward (~ms) is far slower than
//! all workers enqueueing their next sub-batches (~us), so by the time the
//! consumer finishes a forward the queue already holds the next full round ->
//! big batches, GPU stays fed, CPU stays busy.
//!
//! Determinism (the correctness bar): each slot's trajectory depends only on its
//! own rng seed and the (batch-independent, eval-mode) forward outputs — never on
//! thread count, scheduling, or batch composition. So running with n_threads=1
//! vs n_threads=N produces byte-identical games (verified by
//! test_async_mcts_parity.py). The MCTS math is reused verbatim from `mcts.rs`.

use chess_core::board::Board;
use chess_core::mcts::{
    add_dirichlet_noise, backprop, copy_subtree, expand_node, sample_move,
    visits_to_pi, walk_to_leaf, Game, Node, ENC_LEN, MAX_PLIES, POLICY_SIZE,
};
use numpy::ndarray::{Array1, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;

type Row = (Vec<f32>, Vec<f32>, f32); // (state[1152], pi[4672], z)

// --- deterministic per-slot seeding (splitmix64 mixing) ---------------------
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Seed for slot `gi` on its `generation`-th game. Pure function of the global
/// slot index (NOT the worker/shard), so a slot's seed — and thus its whole
/// game trajectory — is independent of how slots are sharded across threads.
fn slot_seed(master: u64, gi: usize, generation: u32) -> u64 {
    let a = (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let b = (generation as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    mix64(master ^ mix64(a ^ b))
}

fn game_uid(global_index: usize, generation: u32) -> u64 {
    (global_index as u64) | ((generation as u64) << 32)
}

// --- per-game slot ----------------------------------------------------------
struct Slot {
    game: Game,
    rng: StdRng,
    global_index: usize,
    generation: u32,
}

impl Slot {
    fn fresh(master_seed: u64, global_index: usize, generation: u32) -> Self {
        Slot {
            game: Game::from_board(Board::new()),
            rng: StdRng::seed_from_u64(slot_seed(master_seed, global_index, generation)),
            global_index,
            generation,
        }
    }
}

// --- worker <-> consumer messages -------------------------------------------
struct EvalRequest {
    rows: Vec<f32>, // n_rows * ENC_LEN, row-major
    n_rows: usize,
    resp: Sender<EvalResponse>,
}
struct EvalResponse {
    logits: Vec<f32>, // n_rows * POLICY_SIZE
    values: Vec<f32>, // n_rows
}
struct FinishedGame {
    #[allow(dead_code)] // carried for result keying / debugging
    game_uid: u64,
    rows: Vec<Row>,
}
enum ToConsumer {
    Eval(EvalRequest),
    Finished(Vec<FinishedGame>),
    WorkerDone,
}

// --- shared immutable config -------------------------------------------------
struct Config {
    sims: usize,
    add_root_noise: bool,
    refill: bool,
    master_seed: u64,
    max_batch: usize,
    stop_flag: AtomicBool,
}

// --- per-game search helpers (mirror MctsEngine, but per-slot) ---------------
fn advance_subtree(g: &mut Game, handle: u16) {
    let child = g.arena[0]
        .children
        .iter()
        .find(|&&(h, _)| h == handle)
        .map(|&(_, c)| c as usize);
    match child {
        Some(ci) => {
            let mut new_arena = Vec::new();
            copy_subtree(&g.arena, ci, &mut new_arena);
            g.arena = new_arena;
        }
        None => {
            g.arena = vec![Node::new(0.0, -1)];
        }
    }
}

fn finalize_rows(g: &mut Game) -> Vec<Row> {
    let z_white = g.result as f32;
    let hist = std::mem::take(&mut g.history);
    hist.into_iter()
        .map(|(state, pi, turn)| {
            let z = if turn { z_white } else { -z_white };
            (state, pi, z)
        })
        .collect()
}

/// One self-play move for a single slot: record (state, pi, turn), sample +
/// push, advance the subtree, mark done on terminal / max-plies. Mirrors
/// MctsEngine::step_moves' per-game body.
fn step_move(slot: &mut Slot) {
    let temp = slot.game.temperature();
    let pi = visits_to_pi(&slot.game.arena, 0, temp);
    let mut state = vec![0.0f32; ENC_LEN];
    slot.game.board.encode_into(&mut state);
    let turn = slot.game.board.turn();
    slot.game.history.push((state, pi, turn));

    let handle = sample_move(&slot.game.arena, 0, temp, &mut slot.rng);
    slot.game.board.push(handle);
    advance_subtree(&mut slot.game, handle);

    let res = slot.game.board.outcome_white();
    if res.is_some() || slot.game.history.len() >= MAX_PLIES {
        slot.game.result = res.unwrap_or(0);
        slot.game.done = true;
    }
}

/// Run one full move-cycle (root expand + `sims` simulations) for every active
/// slot in the shard, offloading each NN batch through `eval`. Leaves stepping
/// the move to the caller (which also finalizes/refills). Mirrors the
/// pending_positions/apply_evals loop of MctsEngine, but over this shard with
/// per-slot rng.
fn run_cycle<F: FnMut(&[f32], usize) -> (Vec<f32>, Vec<f32>)>(
    slots: &mut [Slot],
    sims: usize,
    add_root_noise: bool,
    mut eval: F,
) {
    // ---- ROOT EXPAND (single batch of roots not yet expanded) ----
    let idxs: Vec<usize> = (0..slots.len())
        .filter(|&i| !slots[i].game.done && !slots[i].game.arena[0].expanded)
        .collect();
    if !idxs.is_empty() {
        let mut buf = vec![0.0f32; idxs.len() * ENC_LEN];
        for (row, &si) in idxs.iter().enumerate() {
            slots[si]
                .game
                .board
                .encode_into(&mut buf[row * ENC_LEN..(row + 1) * ENC_LEN]);
        }
        let (logits, _values) = eval(&buf, idxs.len());
        for (row, &si) in idxs.iter().enumerate() {
            let lr = &logits[row * POLICY_SIZE..(row + 1) * POLICY_SIZE];
            let g = &mut slots[si].game;
            expand_node(&mut g.arena, 0, &g.board, lr);
        }
    }
    // root Dirichlet noise (idempotent per node) for all active expanded roots
    if add_root_noise {
        for slot in slots.iter_mut() {
            if !slot.game.done && slot.game.arena[0].expanded {
                add_dirichlet_noise(&mut slot.game.arena, 0, &mut slot.rng);
            }
        }
    }

    // ---- SIMULATIONS ----
    struct LeafCtx {
        slot: usize,
        path: Vec<usize>,
        board: Board,
        eval_row: usize,
    }
    for _s in 0..sims {
        let mut leaves: Vec<LeafCtx> = Vec::new();
        let mut buf: Vec<f32> = Vec::new();
        let mut row = 0usize;
        for si in 0..slots.len() {
            if slots[si].game.done {
                continue;
            }
            let mut board = slots[si].game.board.clone_search();
            let path = walk_to_leaf(&slots[si].game.arena, 0, &mut board);
            match board.terminal_value() {
                Some(t) => {
                    // terminal leaf: backprop immediately (per-slot, no eval)
                    backprop(&mut slots[si].game.arena, &path, t as f64);
                }
                None => {
                    let start = buf.len();
                    buf.resize(start + ENC_LEN, 0.0);
                    board.encode_into(&mut buf[start..start + ENC_LEN]);
                    leaves.push(LeafCtx {
                        slot: si,
                        path,
                        board,
                        eval_row: row,
                    });
                    row += 1;
                }
            }
        }
        if row > 0 {
            let (logits, values) = eval(&buf, row);
            for lf in &leaves {
                let lr = &logits[lf.eval_row * POLICY_SIZE..(lf.eval_row + 1) * POLICY_SIZE];
                let leaf_node = *lf.path.last().unwrap();
                {
                    let g = &mut slots[lf.slot].game;
                    expand_node(&mut g.arena, leaf_node, &lf.board, lr);
                }
                backprop(&mut slots[lf.slot].game.arena, &lf.path, values[lf.eval_row] as f64);
            }
        }
    }
}

/// Worker thread: own a shard of slots, run move-cycles forever (refill) or
/// until the shard drains (no refill), offloading NN batches to the consumer.
fn worker_loop(mut slots: Vec<Slot>, tx: Sender<ToConsumer>, config: Arc<Config>) {
    loop {
        if config.stop_flag.load(Ordering::Relaxed) {
            break;
        }
        if slots.iter().all(|s| s.game.done) {
            break; // no-refill tail: shard exhausted
        }

        // Offload each NN batch through the consumer. On consumer loss, fall
        // back to zeros so we never deadlock (stop_flag will end the loop).
        run_cycle(&mut slots, config.sims, config.add_root_noise, |buf, n| {
            let (resp_tx, resp_rx) = channel();
            let _ = tx.send(ToConsumer::Eval(EvalRequest {
                rows: buf.to_vec(),
                n_rows: n,
                resp: resp_tx,
            }));
            match resp_rx.recv() {
                Ok(r) => (r.logits, r.values),
                Err(_) => (vec![0.0f32; n * POLICY_SIZE], vec![0.0f32; n]),
            }
        });

        // Step one move per active slot; finalize + refill finished games.
        let mut finished: Vec<FinishedGame> = Vec::new();
        for slot in slots.iter_mut() {
            if slot.game.done {
                continue;
            }
            step_move(slot);
            if slot.game.done {
                let uid = game_uid(slot.global_index, slot.generation);
                let rows = finalize_rows(&mut slot.game);
                finished.push(FinishedGame { game_uid: uid, rows });
                if config.refill {
                    *slot = Slot::fresh(config.master_seed, slot.global_index, slot.generation + 1);
                }
            }
        }
        if !finished.is_empty() {
            let _ = tx.send(ToConsumer::Finished(finished));
        }
    }
    let _ = tx.send(ToConsumer::WorkerDone);
}

/// Consumer: the only thread that touches Python. Drain-to-cap batching of NN
/// requests, one big forward per drain, route results back; collect finished
/// games and call `result_sink`; poll `stop`. Returns on shutdown / all workers
/// done / a Python error.
fn consumer_loop(
    rx: Receiver<ToConsumer>,
    config: Arc<Config>,
    n_workers: usize,
    forward: &PyObject,
    result_sink: &PyObject,
    stop: &PyObject,
) -> PyResult<()> {
    let mut workers_done = 0usize;
    loop {
        // Block for the first message of this round.
        let first = match rx.recv() {
            Ok(m) => m,
            Err(_) => break, // all senders dropped
        };

        let mut eval_reqs: Vec<EvalRequest> = Vec::new();
        let mut finished_rows: Vec<Row> = Vec::new();
        let mut n_finished = 0usize;
        let mut total_rows = 0usize;

        let absorb = |msg: ToConsumer,
                          eval_reqs: &mut Vec<EvalRequest>,
                          finished_rows: &mut Vec<Row>,
                          n_finished: &mut usize,
                          total_rows: &mut usize,
                          workers_done: &mut usize| {
            match msg {
                ToConsumer::Eval(req) => {
                    *total_rows += req.n_rows;
                    eval_reqs.push(req);
                }
                ToConsumer::Finished(games) => {
                    for g in games {
                        *n_finished += 1;
                        finished_rows.extend(g.rows);
                    }
                }
                ToConsumer::WorkerDone => *workers_done += 1,
            }
        };
        absorb(first, &mut eval_reqs, &mut finished_rows, &mut n_finished, &mut total_rows, &mut workers_done);

        // Greedily drain everything currently queued, up to the row cap.
        while total_rows < config.max_batch {
            match rx.try_recv() {
                Ok(msg) => absorb(msg, &mut eval_reqs, &mut finished_rows, &mut n_finished, &mut total_rows, &mut workers_done),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // One big forward for all collected eval requests.
        if !eval_reqs.is_empty() {
            let total: usize = eval_reqs.iter().map(|r| r.n_rows).sum();
            let mut batch = Vec::with_capacity(total * ENC_LEN);
            for r in &eval_reqs {
                batch.extend_from_slice(&r.rows);
            }
            let (logits_flat, values_flat) = Python::with_gil(|py| -> PyResult<(Vec<f32>, Vec<f32>)> {
                let arr = Array4::from_shape_vec((total, 18, 8, 8), batch)
                    .unwrap()
                    .into_pyarray(py);
                let out = forward.call1(py, (arr,))?;
                let bound = out.bind(py);
                let (logits, values): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>) =
                    bound.extract()?;
                Ok((logits.as_slice()?.to_vec(), values.as_slice()?.to_vec()))
            })?;
            // Route each request's rows back to its worker.
            let mut off = 0usize;
            for r in eval_reqs {
                let n = r.n_rows;
                let logits = logits_flat[off * POLICY_SIZE..(off + n) * POLICY_SIZE].to_vec();
                let values = values_flat[off..off + n].to_vec();
                off += n;
                let _ = r.resp.send(EvalResponse { logits, values });
            }
        }

        // Deliver finished games to the trainer queue.
        if n_finished > 0 {
            Python::with_gil(|py| -> PyResult<()> {
                let rows: Vec<_> = finished_rows
                    .into_iter()
                    .map(|(s, p, z)| {
                        let s = Array3::from_shape_vec((18, 8, 8), s).unwrap().into_pyarray(py);
                        let p = Array1::from_vec(p).into_pyarray(py);
                        (s, p, z)
                    })
                    .collect();
                let list = PyList::new(py, rows)?;
                result_sink.call1(py, (list, n_finished))?;
                Ok(())
            })?;
        }

        // Poll the Python stop predicate (shutdown / orphan checks live there).
        let should_stop =
            Python::with_gil(|py| -> PyResult<bool> { stop.call0(py)?.bind(py).extract::<bool>() })?;
        if should_stop {
            config.stop_flag.store(true, Ordering::Relaxed);
        }

        // Exit once every worker has reported done. (Under stop we keep serving
        // so any worker blocked on a response unblocks and exits cleanly.)
        if workers_done >= n_workers {
            break;
        }
    }
    Ok(())
}

#[pyclass]
pub struct AsyncMctsEngine {
    n_games: usize,
    sims: usize,
    seed: u64,
    add_root_noise: bool,
    n_threads: usize,
    max_batch: usize,
}

#[pymethods]
impl AsyncMctsEngine {
    #[new]
    fn new(
        n_games: usize,
        sims: usize,
        seed: u64,
        add_root_noise: bool,
        n_threads: usize,
        max_batch: usize,
    ) -> Self {
        AsyncMctsEngine {
            n_games,
            sims,
            seed,
            add_root_noise,
            n_threads: n_threads.max(1),
            max_batch: max_batch.max(1),
        }
    }

    /// Drive the whole multi-threaded self-play loop.
    ///
    /// `forward(batch[M,18,8,8] f32) -> (logits[M,4672] f32, values[M] f32)` is
    /// the only crossing back into Python (the in-process GPU forward).
    /// `result_sink(rows, n_finished)` receives finished (state, pi, z) tuples.
    /// `stop() -> bool` ends the loop (orphan/shutdown checks live there).
    /// refill=True: refill finished slots and run until stop. refill=False: run
    /// until every slot finishes once (the equivalence test path).
    fn run(
        &mut self,
        py: Python<'_>,
        forward: PyObject,
        result_sink: PyObject,
        stop: PyObject,
        refill: bool,
    ) -> PyResult<()> {
        let n_threads = self.n_threads.min(self.n_games).max(1);
        let config = Arc::new(Config {
            sims: self.sims,
            add_root_noise: self.add_root_noise,
            refill,
            master_seed: self.seed,
            max_batch: self.max_batch,
            stop_flag: AtomicBool::new(false),
        });

        // Partition slot indices into `n_threads` contiguous shards.
        let mut shards: Vec<Vec<Slot>> = (0..n_threads).map(|_| Vec::new()).collect();
        for gi in 0..self.n_games {
            let w = gi * n_threads / self.n_games; // contiguous, balanced
            shards[w].push(Slot::fresh(self.seed, gi, 0));
        }
        shards.retain(|s| !s.is_empty());
        let n_workers = shards.len();

        let (tx, rx) = channel::<ToConsumer>();

        // GIL released for the whole threaded section; the consumer reacquires
        // it per batch via Python::with_gil for the forward / sink / stop calls.
        let result = py.allow_threads(move || -> PyResult<()> {
            let mut handles = Vec::with_capacity(n_workers);
            for shard in shards {
                let txc = tx.clone();
                let cfg = config.clone();
                handles.push(thread::spawn(move || worker_loop(shard, txc, cfg)));
            }
            drop(tx); // workers hold the only senders now

            let res = consumer_loop(rx, config.clone(), n_workers, &forward, &result_sink, &stop);
            // Ensure workers exit even if the consumer returned on a Python error
            // (a blocked worker gets a recv error -> zeros -> sees the flag).
            config.stop_flag.store(true, Ordering::Relaxed);
            for h in handles {
                let _ = h.join();
            }
            res
        });
        result
    }
}
