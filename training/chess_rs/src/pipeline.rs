//! Decoupled non-blocking self-play engine (CHESS_BACKEND=rust_pipeline).
//!
//! Replaces the bulk-synchronous `AsyncMctsEngine` with a pipeline where workers
//! NEVER block on the GPU: a worker advances ONE game until it produces exactly
//! one NN leaf, pushes that leaf to an inference bucket, and immediately moves on
//! to another game. So CPU tree-walking overlaps the GPU forward and the GPU is
//! always pre-fed — closing the ~23% GPU bubble the synchronous engine leaves.
//!
//! Topology:
//!   - A GLOBAL priority queue of in-flight games, keyed by ply count (most plies
//!     to the front), so near-complete games drain first and the working set of
//!     expensive search trees stays bounded.
//!   - `n_threads` WORKER threads (GIL released) pop the highest-ply game, apply
//!     any pending NN result (expand+backprop — the workers, not the consumer, do
//!     the tree work), advance the per-game FSM until it yields ONE leaf eval or
//!     completes the game, then pop the next. If the queue is empty a worker
//!     SPAWNS a fresh game (dynamic game count — no fixed `n_games`).
//!   - TWO inference buckets: workers fill one; when it reaches `bucket_size` it is
//!     handed to the consumer and workers fill the other. Flush-on-SIZE only. If
//!     BOTH buckets are full all workers BLOCK (accepted backpressure). In-flight
//!     game count self-regulates to ~2*bucket_size.
//!   - ONE CONSUMER thread: takes a full bucket, runs the Python `forward` under
//!     the GIL back-to-back, and hands the RAW result (numpy, uncopied) to the
//!     scatter thread — it does NO copy, NO re-queue, NO tree work, so forwards run
//!     with no GPU-idle gap between them.
//!   - ONE SCATTER thread: copies the forward output out of numpy ONCE into a
//!     shared `Arc<Vec<f32>>` (under the GIL), then re-queues each game with an
//!     `Arc::clone` + its batch row (queue lock held only for the cheap clones).
//!     This copy overlaps the consumer's NEXT forward (PyTorch frees the GIL during
//!     CUDA compute). Workers slice their own row lazily in `apply_result`.
//!
//! One outstanding NN leaf per game at all times (no virtual loss): a game is
//! EITHER on the queue (no eval) OR in a bucket / being forwarded / being scattered
//! (exactly one eval) — never both. Determinism is NOT a goal (dynamic spawning
//! makes the set of games race-dependent); the routing-integrity test guards
//! correctness.
//!
//! The result rows for a finished game are handed to the Python `submit_game(rows)`
//! callback (brief GIL acquisition; game-rate, so rare). All MCTS math is reused
//! verbatim from `mcts.rs`.

use crate::mcts::{
    add_dirichlet_noise, backprop, copy_subtree, expand_node, sample_move, visits_to_pi,
    walk_to_leaf, Game, Node, ENC_LEN, MAX_PLIES, POLICY_SIZE,
};
use crate::Board;
use numpy::ndarray::{Array1, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use half::bf16;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

type Row = (Vec<f32>, Vec<f32>, f32); // (state[1152], pi[4672], z)

// --- per-game seeding (splitmix64 mixing; determinism not required, just spread) ---
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// --- per-game FSM -----------------------------------------------------------
enum GamePhase {
    NeedRootExpand,    // next: expand root (1 eval, unless root already expanded)
    Simulating(usize), // current simulation index s in 0..sims
}

enum LeafKind {
    RootExpand, // result -> expand_node(root) + dirichlet; no backprop
    SimLeaf,    // result -> expand_node(leaf) + backprop(path, value)
}

/// Search context for the ONE outstanding leaf. `board` is the cloned, walked
/// board at the leaf — it MUST survive the GPU round-trip because `expand_node`
/// needs its legal moves for the priors.
struct LeafCtx {
    kind: LeafKind,
    path: Vec<usize>, // arena indices root..leaf (for backprop). Root: [0].
    board: Board,
}

struct EvalResult {
    // The FULL batch policy [m * POLICY_SIZE], shared by all games in the batch.
    // The worker slices its own `row` lazily in `apply_result` (GIL-free) — no
    // per-game copy on the consumer's GPU-launch path.
    logits: Arc<Vec<f32>>,
    row: usize, // this game's row in the batch
    value: f32,
}

/// A game in flight. Boxed so it moves queue<->bucket<->worker by pointer.
struct InFlight {
    id: u64,
    game: Game,
    rng: StdRng,
    phase: GamePhase,
    pending: Option<EvalResult>, // written by the scatter thread, consumed by the worker
    leaf_ctx: Option<LeafCtx>,   // Some exactly while an eval is outstanding
    staged_enc: Vec<f32>,        // encoded leaf board, copied into the bucket on push
}

fn spawn_fresh(shared: &Shared) -> InFlight {
    let id = shared.id_counter.fetch_add(1, AtomicOrdering::Relaxed);
    let seed = mix64(shared.config.seed ^ mix64(id.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    InFlight {
        id,
        game: Game::from_board(Board::new()),
        rng: StdRng::seed_from_u64(seed),
        phase: GamePhase::NeedRootExpand,
        pending: None,
        leaf_ctx: None,
        staged_enc: Vec::new(),
    }
}

// --- priority queue entry (max-heap by ply, id as total-order tie-break) -----
struct Queued {
    key: u64, // history.len(): most plies pops first
    seq: u64, // id
    item: Box<InFlight>,
}
impl PartialEq for Queued {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.seq == o.seq
    }
}
impl Eq for Queued {}
impl Ord for Queued {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key.cmp(&o.key).then_with(|| self.seq.cmp(&o.seq))
    }
}
impl PartialOrd for Queued {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// --- two-bucket inference staging -------------------------------------------
struct Bucket {
    enc: Vec<f32>,            // len = games.len() * ENC_LEN, row-major
    games: Vec<Box<InFlight>>, // row r's game travels WITH the bucket (positional routing)
}
impl Bucket {
    fn with_capacity(b: usize) -> Self {
        Bucket {
            enc: Vec::with_capacity(b * ENC_LEN),
            games: Vec::with_capacity(b),
        }
    }
    fn len(&self) -> usize {
        self.games.len()
    }
}

struct BucketState {
    active: Bucket,         // workers append here
    ready: Option<Bucket>,  // a full bucket waiting for the consumer (2nd buffer)
}

// --- zero-copy host I/O ring (GB10/ATS) --------------------------------------
// When `Shared.ring` is Some, the pipeline runs the ZERO-COPY path instead of
// the numpy buckets above: workers write encoded leaves (as bf16) DIRECTLY into
// a slot's pinned host input buffer that the GPU reads in place via ATS; the net
// writes bf16 outputs into the slot's pinned output buffers; the scatter thread
// reads them after the slot's CUDA event. Buffers are owned by Python
// (zerocopy.py); here we hold only the raw host pointers (as usize -> Send+Sync,
// cast at use sites). A slot is in use from the moment a worker starts filling
// it until the scatter thread has finished reading its outputs, at which point it
// returns to `free_slots` (this is the event-gated reuse guarantee: a slot is
// never re-filled while its outputs are still being read or its forward is live).
struct HostRing {
    input: Vec<usize>,   // n raw ptrs to pinned bf16 [B, in_stride]
    logit: Vec<usize>,   // n raw ptrs to pinned bf16 [B, POLICY_SIZE]
    value: Vec<usize>,   // n raw ptrs to pinned bf16 [B]
    in_stride: usize,    // ENC_LEN (1152); logit row stride is POLICY_SIZE
}

struct RingBucket {
    slot: usize,
    games: Vec<Box<InFlight>>, // row r's game travels WITH the slot (positional)
}

struct RingState {
    active: RingBucket,           // workers fill this slot's input buffer
    ready: VecDeque<RingBucket>,  // full buckets waiting for the consumer
}

struct Config {
    sims: usize,
    add_root_noise: bool,
    bucket_size: usize,
    seed: u64,
}

struct Shared {
    queue: Mutex<BinaryHeap<Queued>>,
    buckets: Mutex<BucketState>,
    bucket_cv_worker: Condvar,   // workers wait here when both buckets full
    bucket_cv_consumer: Condvar, // consumer waits here for a full bucket
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
    // zero-copy ring (Some => ring path; legacy bucket fields above unused)
    ring: Option<HostRing>,
    ring_state: Mutex<RingState>,
    ring_consumer_cv: Condvar,        // consumer waits here for a ready slot
    free_slots: Mutex<Vec<usize>>,    // slots not currently in use
    slot_cv: Condvar,                 // workers wait here for a free slot
    sync_slot: Option<PyObject>,      // Python ring.sync(slot): waits the slot event
}

// --- per-game helpers (mirror MctsEngine/async_mcts, but standalone so the
//     rust_async baseline stays untouched) ---------------------------------
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

/// Pure-CPU move step (no NN eval): record (state, pi, turn), sample + push,
/// advance the subtree, mark done on terminal / max-plies.
fn step_move(g: &mut InFlight) {
    let temp = g.game.temperature();
    let pi = visits_to_pi(&g.game.arena, 0, temp);
    let mut state = vec![0.0f32; ENC_LEN];
    g.game.board.encode_into(&mut state);
    let turn = g.game.board.turn();
    g.game.history.push((state, pi, turn));

    let handle = sample_move(&g.game.arena, 0, temp, &mut g.rng);
    g.game.board.push(handle);
    advance_subtree(&mut g.game, handle);

    let res = g.game.board.outcome_white();
    if res.is_some() || g.game.history.len() >= MAX_PLIES {
        g.game.result = res.unwrap_or(0);
        g.game.done = true;
    }
}

/// Apply the consumer-attached NN result to the tree (workers do this, not the
/// consumer), then advance the FSM cursor.
fn apply_result(g: &mut InFlight, res: EvalResult, config: &Config) {
    let ctx = g.leaf_ctx.take().expect("pending result without leaf_ctx");
    // Lazy slice into the shared batch buffer — this game's row only.
    let logits = &res.logits[res.row * POLICY_SIZE..(res.row + 1) * POLICY_SIZE];
    match ctx.kind {
        LeafKind::RootExpand => {
            expand_node(&mut g.game.arena, 0, &ctx.board, logits);
            if config.add_root_noise {
                add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
            }
            g.phase = GamePhase::Simulating(0);
        }
        LeafKind::SimLeaf => {
            let leaf = *ctx.path.last().unwrap();
            expand_node(&mut g.game.arena, leaf, &ctx.board, logits);
            backprop(&mut g.game.arena, &ctx.path, res.value as f64);
            if let GamePhase::Simulating(s) = g.phase {
                g.phase = GamePhase::Simulating(s + 1);
            }
        }
    }
}

enum Advance {
    Eval,            // g.leaf_ctx is Some, g.staged_enc holds the encoded board
    Done(Vec<Row>),  // game completed; rows to submit
}

/// Advance the game until it produces ONE leaf eval or completes. Multiple
/// terminal sims (no eval) may be processed in one call.
fn advance_until_eval_or_done(g: &mut InFlight, config: &Config) -> Advance {
    loop {
        match g.phase {
            GamePhase::NeedRootExpand => {
                if g.game.arena[0].expanded {
                    // reused subtree root: no eval, just (idempotent) root noise
                    if config.add_root_noise {
                        add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
                    }
                    g.phase = GamePhase::Simulating(0);
                    continue;
                }
                let board = g.game.board.clone_search();
                let mut buf = vec![0.0f32; ENC_LEN];
                board.encode_into(&mut buf);
                g.staged_enc = buf;
                g.leaf_ctx = Some(LeafCtx {
                    kind: LeafKind::RootExpand,
                    path: vec![0],
                    board,
                });
                return Advance::Eval;
            }
            GamePhase::Simulating(s) => {
                if s >= config.sims {
                    step_move(g);
                    if g.game.done {
                        return Advance::Done(finalize_rows(&mut g.game));
                    }
                    g.phase = GamePhase::NeedRootExpand;
                    continue;
                }
                let mut board = g.game.board.clone_search();
                let path = walk_to_leaf(&g.game.arena, 0, &mut board);
                match board.terminal_value() {
                    Some(t) => {
                        backprop(&mut g.game.arena, &path, t as f64);
                        g.phase = GamePhase::Simulating(s + 1);
                        continue;
                    }
                    None => {
                        let mut buf = vec![0.0f32; ENC_LEN];
                        board.encode_into(&mut buf);
                        g.staged_enc = buf;
                        g.leaf_ctx = Some(LeafCtx {
                            kind: LeafKind::SimLeaf,
                            path,
                            board,
                        });
                        return Advance::Eval;
                    }
                }
            }
        }
    }
}

/// Push a staged game into the active bucket. Blocks only when BOTH buckets are
/// full (backpressure). On shutdown, drops the game.
fn push_to_bucket(shared: &Shared, g: Box<InFlight>) {
    let b = shared.config.bucket_size;
    let mut bs = shared.buckets.lock().unwrap();
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return; // drop g
        }
        if bs.active.len() < b {
            bs.active.enc.extend_from_slice(&g.staged_enc);
            bs.active.games.push(g);
            if bs.active.len() == b && bs.ready.is_none() {
                let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
                bs.ready = Some(full);
                shared.bucket_cv_consumer.notify_one();
            }
            return;
        }
        // active is full
        if bs.ready.is_none() {
            let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
            bs.ready = Some(full);
            shared.bucket_cv_consumer.notify_one();
            continue; // active now empty -> append on next iteration
        }
        // both buckets full -> block until the consumer takes `ready`
        bs = shared.bucket_cv_worker.wait(bs).unwrap();
    }
}

// --- zero-copy ring worker path ---------------------------------------------
/// Write one game's encoded leaf (f32 `enc`) into slot `slot`'s pinned host input
/// buffer at `row`, converting to bf16 in place. SAFETY: callers hold the
/// `ring_state` lock, so `slot` is the active slot and only one worker writes a
/// given (slot,row) at a time; the GPU does not read this slot until it is full
/// and handed to the consumer.
fn write_input_row(ring: &HostRing, slot: usize, row: usize, enc: &[f32]) {
    debug_assert_eq!(enc.len(), ring.in_stride);
    let dst = ring.input[slot] as *mut u16;
    unsafe {
        let base = dst.add(row * ring.in_stride);
        for (i, &x) in enc.iter().enumerate() {
            *base.add(i) = bf16::from_f32(x).to_bits();
        }
    }
}

/// Pop a free slot, blocking on `slot_cv` until one is available. Returns None on
/// shutdown. MUST be called WITHOUT holding `ring_state` (deadlock-safe): the
/// consumer needs `ring_state` to drain `ready`, which lets scatter free a slot.
fn pop_free_slot_blocking(shared: &Shared) -> Option<usize> {
    let mut fs = shared.free_slots.lock().unwrap();
    loop {
        if let Some(s) = fs.pop() {
            return Some(s);
        }
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return None;
        }
        fs = shared.slot_cv.wait(fs).unwrap();
    }
}

/// Ring analogue of `push_to_bucket`: append `g` to the active slot, rotating to a
/// fresh slot when the active one fills. Backpressure is "no free slot" (workers
/// block in `pop_free_slot_blocking`, never while holding `ring_state`).
fn push_ring(shared: &Shared, g: Box<InFlight>) {
    let b = shared.config.bucket_size;
    let ring = shared.ring.as_ref().unwrap();
    let mut g = Some(g);
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return; // drop g
        }
        let mut rs = shared.ring_state.lock().unwrap();
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        if rs.active.games.len() < b {
            let slot = rs.active.slot;
            let row = rs.active.games.len();
            let gg = g.take().unwrap();
            write_input_row(ring, slot, row, &gg.staged_enc);
            rs.active.games.push(gg);
            if rs.active.games.len() == b {
                // active full: rotate to ready + start a new active on a fresh
                // slot. Try non-blocking first (don't block holding ring_state).
                let ns = shared.free_slots.lock().unwrap().pop();
                if let Some(ns) = ns {
                    let full = std::mem::replace(
                        &mut rs.active,
                        RingBucket { slot: ns, games: Vec::with_capacity(b) });
                    rs.ready.push_back(full);
                    shared.ring_consumer_cv.notify_one();
                }
                // else: no free slot now; leave active full (unflushed). The next
                // push lands in the active-full branch below and flushes after
                // blocking for a slot OUTSIDE this lock. The consumer keeps
                // draining `ready` meanwhile, so a slot will free up.
            }
            return;
        }
        // active is full and was not rotated (no free slot earlier). Flush it now:
        // acquire a slot WITHOUT holding ring_state, then rotate.
        drop(rs);
        let ns = match pop_free_slot_blocking(shared) {
            Some(s) => s,
            None => return, // shutting down
        };
        {
            let mut rs = shared.ring_state.lock().unwrap();
            if rs.active.games.len() == b {
                let full = std::mem::replace(
                    &mut rs.active,
                    RingBucket { slot: ns, games: Vec::with_capacity(b) });
                rs.ready.push_back(full);
                shared.ring_consumer_cv.notify_one();
                // active now empty; loop to append g
            } else {
                // another worker already rotated; return our slot and retry
                let mut fs = shared.free_slots.lock().unwrap();
                fs.push(ns);
                shared.slot_cv.notify_one();
            }
        }
        // loop continues; g still Some
    }
}

/// Hand a finished game's rows to the Python `submit_game(rows)` callback.
fn submit_completed(submit_game: &PyObject, rows: Vec<Row>) {
    if rows.is_empty() {
        return;
    }
    let _ = Python::with_gil(|py| -> PyResult<()> {
        let pyrows: Vec<_> = rows
            .into_iter()
            .map(|(s, p, z)| {
                let s = Array3::from_shape_vec((18, 8, 8), s).unwrap().into_pyarray(py);
                let p = Array1::from_vec(p).into_pyarray(py);
                (s, p, z)
            })
            .collect();
        let list = PyList::new(py, pyrows)?;
        submit_game.call1(py, (list,))?;
        Ok(())
    });
}

/// Worker: pop (or spawn) a game, apply any pending result, advance until it
/// yields one eval (push to bucket) or completes (submit), forever until stop.
fn worker_loop(shared: Arc<Shared>, submit_game: PyObject) {
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }

        // (A) pop the highest-ply game, or spawn a fresh one if the queue is empty.
        let mut g: Box<InFlight> = {
            let mut q = shared.queue.lock().unwrap();
            match q.pop() {
                Some(qd) => qd.item,
                None => {
                    drop(q);
                    Box::new(spawn_fresh(&shared))
                }
            }
        };

        // (B) apply the consumer-attached result, if any.
        if let Some(res) = g.pending.take() {
            apply_result(&mut g, res, &shared.config);
        }

        // (C) advance the FSM until it produces one eval or completes.
        match advance_until_eval_or_done(&mut g, &shared.config) {
            Advance::Eval => {
                if shared.ring.is_some() {
                    push_ring(&shared, g);
                } else {
                    push_to_bucket(&shared, g);
                }
            }
            Advance::Done(rows) => submit_completed(&submit_game, rows),
        }
    }
}

/// Handoff from the consumer to the scatter thread, positionally tied to its m
/// games. Legacy: the raw Python `forward` result (logits[m,POLICY] + values[m]
/// numpy, held as a Py object — `Send`, needs the GIL only to read). Ring: the
/// slot index whose pinned bf16 output buffers hold the result (read after the
/// slot's CUDA event via `sync_slot`).
enum ScatterMsg {
    Numpy(PyObject, Vec<Box<InFlight>>),
    Slot(usize, Vec<Box<InFlight>>),
}

/// Scatter thread: OFF the consumer's GPU-launch path. Copy the forward output
/// out of numpy ONCE into an `Arc<Vec<f32>>` (under the GIL — numpy can't outlive
/// it), then DROP the GIL and re-queue each game with a cheap `Arc::clone` + its
/// row index. The queue lock is held only for the clones+pushes (microseconds),
/// not the ~1.3 ms of memcpy it used to be — so workers no longer stall on it and
/// the consumer's next forward overlaps this copy (PyTorch frees the GIL during
/// CUDA compute). Workers slice their own row lazily in `apply_result`.
/// Re-queue each scattered game with a cheap `Arc::clone` of the shared policy +
/// its row index/value. Queue lock held only for the clones+pushes (microseconds).
/// Workers slice their own row lazily in `apply_result`.
fn requeue_scattered(
    shared: &Shared,
    logits: Arc<Vec<f32>>,
    values: &[f32],
    games: Vec<Box<InFlight>>,
) {
    let mut q = shared.queue.lock().unwrap();
    for (r, mut g) in games.into_iter().enumerate() {
        g.pending = Some(EvalResult {
            logits: Arc::clone(&logits),
            row: r,
            value: values[r],
        });
        let key = g.game.history.len() as u64;
        let seq = g.id;
        q.push(Queued { key, seq, item: g });
    }
}

fn scatter_loop(shared: Arc<Shared>, rx: Receiver<ScatterMsg>) {
    // Exits when the consumer drops the sender (recv -> Err) — i.e. on shutdown.
    while let Ok(msg) = rx.recv() {
        match msg {
            // --- legacy numpy path ---
            ScatterMsg::Numpy(result, games) => {
                // Hold the GIL only to validate the arrays and grab raw buffer
                // pointers (microseconds); the big memcpy then runs GIL-RELEASED,
                // overlapping the consumer's next forward.
                let ptrs =
                    Python::with_gil(|py| -> PyResult<(*const f32, usize, *const f32, usize)> {
                        let bound = result.bind(py);
                        let (logits, values): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>) =
                            bound.extract()?;
                        let l = logits.as_slice()?; // validates C-contiguity
                        let v = values.as_slice()?;
                        Ok((l.as_ptr(), l.len(), v.as_ptr(), v.len()))
                    });
                let (lptr, llen, vptr, vlen) = match ptrs {
                    Ok(p) => p,
                    Err(e) => {
                        // Unexpected shape/type: fail the run rather than silently
                        // mis-route corrupted targets into training.
                        Python::with_gil(|py| e.print(py));
                        shared.stop.store(true, AtomicOrdering::Relaxed);
                        shared.bucket_cv_worker.notify_all();
                        shared.bucket_cv_consumer.notify_all();
                        return;
                    }
                };
                // SAFETY: `result` (dropped below) keeps the numpy arrays + buffers
                // alive; numpy never relocates a buffer and nothing mutates these
                // fresh arrays. GIL released here -> overlaps the next forward.
                let logits_flat = unsafe { std::slice::from_raw_parts(lptr, llen) }.to_vec();
                let values_flat = unsafe { std::slice::from_raw_parts(vptr, vlen) }.to_vec();
                Python::with_gil(|_py| drop(result));
                requeue_scattered(&shared, Arc::new(logits_flat), &values_flat, games);
            }
            // --- zero-copy ring path ---
            ScatterMsg::Slot(slot, games) => {
                // Wait the slot's CUDA event (GIL released inside torch) before
                // reading its pinned output buffers. Replaces the implicit .cpu()
                // sync. SAFETY for the reads below: after this returns, the GPU has
                // finished writing this slot's outputs, and the slot will not be
                // re-filled until we return it to free_slots at the end.
                let sync = shared.sync_slot.as_ref().unwrap();
                let synced = Python::with_gil(|py| -> PyResult<()> {
                    sync.call1(py, (slot,))?;
                    Ok(())
                });
                if let Err(e) = synced {
                    Python::with_gil(|py| e.print(py));
                    shared.stop.store(true, AtomicOrdering::Relaxed);
                    shared.ring_consumer_cv.notify_all();
                    shared.slot_cv.notify_all();
                    return;
                }
                let ring = shared.ring.as_ref().unwrap();
                let m = games.len();
                let lbase = ring.logit[slot] as *const u16;
                let vbase = ring.value[slot] as *const u16;
                let mut logits_flat = Vec::with_capacity(m * POLICY_SIZE);
                let mut values_flat = Vec::with_capacity(m);
                // bf16 -> f32 (the net ran bf16, so this loses nothing the f32 path
                // had). Off the consumer's GPU-launch path, overlaps next forward.
                unsafe {
                    for i in 0..m * POLICY_SIZE {
                        logits_flat.push(bf16::from_bits(*lbase.add(i)).to_f32());
                    }
                    for r in 0..m {
                        values_flat.push(bf16::from_bits(*vbase.add(r)).to_f32());
                    }
                }
                requeue_scattered(&shared, Arc::new(logits_flat), &values_flat, games);
                // Return the slot AFTER its outputs are fully read (event-gated
                // reuse: a slot is never re-filled while still in use).
                shared.free_slots.lock().unwrap().push(slot);
                shared.slot_cv.notify_one();
            }
        }
    }
}

/// Consumer: the only thread running the GPU forward. Take a full bucket, run
/// `forward` back-to-back, and hand the raw result to the scatter thread (which
/// copies it out + re-queues). Polls `stop` (with a bounded wait so it still polls
/// when no bucket ever fills — flush-on-size-only is preserved).
/// Consumer for the zero-copy ring: take a ready slot, call `forward(slot, m)`
/// (runs the net in place on the slot's pinned input, writes bf16 outputs into
/// its pinned output buffers, records the slot's event), and hand the slot to
/// scatter. No numpy, no result copy on this thread.
fn consumer_ring(
    shared: Arc<Shared>,
    forward: PyObject,
    stop: PyObject,
    scatter_tx: SyncSender<ScatterMsg>,
) -> PyResult<()> {
    // The post-forward stop() callback (getppid + an Event semaphore acquire) is
    // throttled to ~100ms so it isn't paid on every forward's hot path; the cheap
    // `shared.stop` atomic is still checked every iteration in (A).
    let mut last_poll = std::time::Instant::now();
    loop {
        // (A) acquire a ready slot, polling stop() while idle.
        let rb = loop {
            {
                let rs = shared.ring_state.lock().unwrap();
                if shared.stop.load(AtomicOrdering::Relaxed) {
                    return Ok(());
                }
                let mut rs = rs;
                if let Some(rb) = rs.ready.pop_front() {
                    break rb;
                }
                let _ = shared
                    .ring_consumer_cv
                    .wait_timeout(rs, Duration::from_millis(250))
                    .unwrap();
            }
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return Ok(());
            }
            if poll_stop(&shared, &stop)? {
                return Ok(());
            }
        };
        // (B) forward(slot, m) under the GIL.
        let RingBucket { slot, games } = rb;
        let m = games.len();
        Python::with_gil(|py| -> PyResult<()> {
            forward.call1(py, (slot, m))?;
            Ok(())
        })?;
        // (C) hand slot + games to scatter; loop straight into the next forward.
        if scatter_tx.send(ScatterMsg::Slot(slot, games)).is_err() {
            shared.stop.store(true, AtomicOrdering::Relaxed);
            return Ok(());
        }
        // (D) poll stop after the forward (throttled; see last_poll above).
        if last_poll.elapsed() >= Duration::from_millis(100) {
            last_poll = std::time::Instant::now();
            if poll_stop(&shared, &stop)? {
                return Ok(());
            }
        }
    }
}

fn consumer_loop(
    shared: Arc<Shared>,
    forward: PyObject,
    stop: PyObject,
    scatter_tx: SyncSender<ScatterMsg>,
) -> PyResult<()> {
    if shared.ring.is_some() {
        return consumer_ring(shared, forward, stop, scatter_tx);
    }
    let mut last_poll = std::time::Instant::now(); // throttle the post-forward stop()
    loop {
        // (A) acquire a full bucket, polling stop() while idle.
        let bucket = loop {
            {
                let mut bs = shared.buckets.lock().unwrap();
                if shared.stop.load(AtomicOrdering::Relaxed) {
                    return Ok(());
                }
                if let Some(b) = bs.ready.take() {
                    shared.bucket_cv_worker.notify_all(); // a blocked worker can now promote
                    break b;
                }
                let _ = shared
                    .bucket_cv_consumer
                    .wait_timeout(bs, Duration::from_millis(250))
                    .unwrap();
            }
            // lock released; poll stop() (a never-filling bucket must not wedge us).
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return Ok(());
            }
            if poll_stop(&shared, &stop)? {
                return Ok(());
            }
        };

        // (B) GPU forward under the GIL. Keep the raw numpy result — do NOT copy
        // it out here; that copy belongs off this thread (the scatter thread).
        let Bucket { enc, games } = bucket;
        let m = games.len();
        let result: PyObject = Python::with_gil(|py| -> PyResult<PyObject> {
            let arr = Array4::from_shape_vec((m, 18, 8, 8), enc)
                .unwrap()
                .into_pyarray(py);
            forward.call1(py, (arr,))
        })?;

        // (C) hand the result + its games to the scatter thread and loop straight
        // back into the next forward. Bounded channel -> backpressure if scatter
        // falls behind. A send error means scatter has exited (shutdown).
        if scatter_tx.send(ScatterMsg::Numpy(result, games)).is_err() {
            shared.stop.store(true, AtomicOrdering::Relaxed);
            return Ok(());
        }

        // (D) poll stop after the forward (throttled; see last_poll above).
        if last_poll.elapsed() >= Duration::from_millis(100) {
            last_poll = std::time::Instant::now();
            if poll_stop(&shared, &stop)? {
                return Ok(());
            }
        }
    }
}

fn poll_stop(shared: &Shared, stop: &PyObject) -> PyResult<bool> {
    let should = Python::with_gil(|py| -> PyResult<bool> {
        stop.call0(py)?.bind(py).extract::<bool>()
    })?;
    if should {
        shared.stop.store(true, AtomicOrdering::Relaxed);
        shared.bucket_cv_worker.notify_all();
        shared.bucket_cv_consumer.notify_all();
    }
    Ok(should)
}

#[pyclass]
pub struct PipelineEngine {
    sims: usize,
    seed: u64,
    add_root_noise: bool,
    n_threads: usize,
    bucket_size: usize,
}

#[pymethods]
impl PipelineEngine {
    #[new]
    fn new(sims: usize, seed: u64, add_root_noise: bool, n_threads: usize, bucket_size: usize) -> Self {
        PipelineEngine {
            sims,
            seed,
            add_root_noise,
            n_threads: n_threads.max(1),
            bucket_size: bucket_size.max(1),
        }
    }

    /// Drive the decoupled self-play pipeline until `stop()` is true.
    ///
    /// `forward(batch[M,18,8,8] f32) -> (logits[M,4672] f32, values[M] f32)` — the
    /// in-process GPU forward (consumer-only). `submit_game(rows)` receives one
    /// finished game's list of (state[18,8,8], pi[4672], z) tuples. `stop() -> bool`
    /// ends the run (orphan/shutdown checks live there). In-flight games are
    /// dropped on shutdown (no drain).
    /// Ring params (zero-copy path) are optional: passing `input_ptrs` (one pinned
    /// host bf16 input buffer per slot) + `logit_ptrs`/`value_ptrs` + `sync_slot`
    /// switches `forward` to the `forward(slot_idx, m)` contract. Omit them (the
    /// routing test does) to keep the legacy `forward(batch)->(logits,values)` path.
    #[pyo3(signature = (forward, submit_game, stop, sync_slot=None,
                        input_ptrs=Vec::new(), logit_ptrs=Vec::new(),
                        value_ptrs=Vec::new(), n_slots=0, in_stride=0,
                        logit_stride=0))]
    fn run(
        &mut self,
        py: Python<'_>,
        forward: PyObject,
        submit_game: PyObject,
        stop: PyObject,
        sync_slot: Option<PyObject>,
        input_ptrs: Vec<usize>,
        logit_ptrs: Vec<usize>,
        value_ptrs: Vec<usize>,
        n_slots: usize,
        in_stride: usize,
        logit_stride: usize,
    ) -> PyResult<()> {
        let n_threads = self.n_threads.max(1);
        let bucket_size = self.bucket_size.max(1);
        let ring = if !input_ptrs.is_empty() {
            // scatter reads logit rows with POLICY_SIZE stride (matches apply_result)
            assert_eq!(logit_stride, POLICY_SIZE, "ring logit stride must == POLICY_SIZE");
            Some(HostRing {
                input: input_ptrs,
                logit: logit_ptrs,
                value: value_ptrs,
                in_stride,
            })
        } else {
            None
        };
        // Ring free-list: slot 0 is bound to the initial active bucket; 1..n free.
        let free_init: Vec<usize> = if ring.is_some() {
            (1..n_slots).collect()
        } else {
            Vec::new()
        };
        let shared = Arc::new(Shared {
            queue: Mutex::new(BinaryHeap::new()),
            buckets: Mutex::new(BucketState {
                active: Bucket::with_capacity(bucket_size),
                ready: None,
            }),
            bucket_cv_worker: Condvar::new(),
            bucket_cv_consumer: Condvar::new(),
            stop: AtomicBool::new(false),
            id_counter: AtomicU64::new(0),
            config: Config {
                sims: self.sims,
                add_root_noise: self.add_root_noise,
                bucket_size,
                seed: self.seed,
            },
            ring,
            ring_state: Mutex::new(RingState {
                active: RingBucket {
                    slot: 0,
                    games: Vec::with_capacity(bucket_size),
                },
                ready: VecDeque::new(),
            }),
            ring_consumer_cv: Condvar::new(),
            free_slots: Mutex::new(free_init),
            slot_cv: Condvar::new(),
            sync_slot,
        });

        // Clone the submit_game callable per worker while we still hold the GIL.
        let submit_games: Vec<PyObject> =
            (0..n_threads).map(|_| submit_game.clone_ref(py)).collect();

        // GIL released for the threaded section; consumer/workers reacquire it per
        // forward / submit_game / stop via Python::with_gil.
        py.allow_threads(move || -> PyResult<()> {
            // Consumer -> scatter handoff. Depth 2 lets the consumer run a forward
            // (or two) ahead of the scatter copy; combined with the two-bucket
            // staging this is the pipeline's double-buffering.
            let (scatter_tx, scatter_rx) = sync_channel::<ScatterMsg>(2);
            let mut handles = Vec::with_capacity(n_threads);
            for sg in submit_games {
                let sh = shared.clone();
                handles.push(thread::spawn(move || worker_loop(sh, sg)));
            }
            let scatter_handle = {
                let sh = shared.clone();
                thread::spawn(move || scatter_loop(sh, scatter_rx))
            };
            // consumer_loop owns scatter_tx; when it returns the sender drops, so
            // the scatter thread sees recv() -> Err and exits.
            let res = consumer_loop(shared.clone(), forward, stop, scatter_tx);
            // Ensure workers exit even on a Python error / early return.
            shared.stop.store(true, AtomicOrdering::Relaxed);
            shared.bucket_cv_worker.notify_all();
            shared.bucket_cv_consumer.notify_all();
            shared.ring_consumer_cv.notify_all();
            shared.slot_cv.notify_all();
            for h in handles {
                let _ = h.join();
            }
            let _ = scatter_handle.join();
            res
        })
    }
}
