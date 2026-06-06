//! Decoupled non-blocking self-play pipeline, 100% Rust (no GIL, no PyO3).
//!
//! Port of the Rust half of the old `chess_rs/src/pipeline.rs` with the entire
//! Python-boundary apparatus removed: the consumer/scatter split, the
//! `Arc<Vec<f32>>` copy-out under the GIL, the numpy round-trip. Here ONE
//! inference thread owns the tch net + a CUDA-graphed static device slot and
//! talks to the worker threads with plain Rust data.
//!
//! Topology:
//!   - global ply-priority queue of in-flight games (most plies first);
//!   - N worker threads advance a game's FSM until it yields ONE leaf eval (push
//!     to the leaf channel) or completes (hand rows to the sink); empty queue =>
//!     spawn a fresh game (dynamic game count, self-regulates to fill the pipe).
//!     Workers also do the dtype conversions on both edges of the forward — they
//!     narrow the leaf encoding f32->bf16 before pushing, and widen their own
//!     output row bf16->f32 after — so the two singleton GPU-feeder threads stay
//!     thin and the per-element work is parallelized across all N workers. The
//!     16-bit payload is carried host-side as raw bf16 bits in an i16 (tch 0.23
//!     mis-types half::bf16 as Kind::Half, so we bitcast via Tensor::view_dtype
//!     at the device boundary rather than relying on its bf16 Element);
//!   - ONE bounded ply-priority pre-inference queue (Mutex<BinaryHeap> + condvars,
//!     cap = 2*n_slots*BATCH): workers push and block only at capacity
//!     (backpressure); the inference thread pops the highest-ply games first, so
//!     the games closest to finishing are evaluated soonest (throughput);
//!   - ONE inference thread: gathers BATCH bf16 leaves off the channel, memcpys
//!     them into a free pinned bf16 device slot, launches the fused AOTI forward,
//!     and hands the in-flight slot to the scatter thread;
//!   - ONE scatter thread: waits the forward's event, copies the bf16 outputs back
//!     (D2H only, no widening), requeues the games with results, recycles the slot.
//!
//! Weight hot-reload: the inference thread mtime-polls the safetensors sidecar
//! and reloads IN PLACE (Net::reload), so the captured graph keeps replaying the
//! fresh weights with no recapture.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ffi::CString;
use std::io::{self, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use half::bf16;

use chess_core::board::Board;
use chess_core::mcts::{
    add_dirichlet_noise, backprop, copy_subtree, expand_node, sample_move, visits_to_pi,
    walk_to_leaf, Game, Node, ENC_LEN, MAX_PLIES, POLICY_SIZE,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use crate::aoti::AotiModel;
use crate::cudagraph::{self, CudaEvent};

// GPU batch width per forward. The serving_model.pt2 is exported with a STATIC
// batch (no dynamic_shapes), so this MUST equal SERVING_BATCH in training/net.py —
// enforced at startup by AotiModel::check_batch (aoti_check_batch in shim.cpp).
pub const BATCH: usize = 1024;

// In-flight forwards (= pinned input slots = max GPU forwards queued ahead) is
// configurable via Config::n_slots (--slots); more slots hide scatter/CPU jitter
// so the GPU stays fed (higher duty cycle). Slot size is fixed at BATCH.

pub type Row = (Vec<f32>, Vec<f32>, f32); // (state[1152], pi[4672], z)

// --- per-game seeding (splitmix64; determinism not required, just spread) ---
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// --- per-game FSM (verbatim from the original Rust pipeline) ------------------
enum GamePhase {
    NeedRootExpand,
    Simulating(usize),
}
enum LeafKind {
    RootExpand,
    SimLeaf,
}
struct LeafCtx {
    kind: LeafKind,
    path: Vec<usize>,
    board: Board,
}
struct EvalResult {
    logits: Arc<Vec<i16>>, // full batch policy [m * POLICY_SIZE] as raw bf16 bits (widened per-row by the worker)
    row: usize,
    value: i16, // this game's value as raw bf16 bits (widened by the worker)
}
struct InFlight {
    id: u64,
    game: Game,
    rng: StdRng,
    phase: GamePhase,
    pending: Option<EvalResult>,
    leaf_ctx: Option<LeafCtx>,
    staged_enc: Vec<i16>, // board encoding, already narrowed to bf16 (raw bits in i16) by the worker
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

struct Queued {
    key: u64,
    seq: u64,
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

/// Bounded, blocking ply-priority queue feeding the inference thread (the
/// "pre-inference" queue). Same ordering as the global worker queue — most plies
/// first via `Queued` — but bounded at `cap` with a blocking push (backpressure)
/// and a `gather` that waits for a full BATCH, i.e. the crossbeam leaf channel's
/// blocking / backpressure / disconnect semantics PLUS priority ordering so the
/// games closest to finishing are evaluated first. `closed` + `notify_all` on
/// both condvars replaces the channel's drop-based disconnect for shutdown.
struct PreInferQueue {
    inner: Mutex<BinaryHeap<Queued>>,
    not_empty: Condvar,
    not_full: Condvar,
    cap: usize,
    closed: AtomicBool,
}

impl PreInferQueue {
    fn new(cap: usize) -> PreInferQueue {
        PreInferQueue {
            inner: Mutex::new(BinaryHeap::new()),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            cap,
            closed: AtomicBool::new(false),
        }
    }

    /// Worker push. Blocks while the queue is at capacity (backpressure). Returns
    /// `Err(())` once the queue is closed (shutdown), mirroring the old leaf
    /// channel's `send` error when the receiver was gone.
    fn push(&self, g: Box<InFlight>) -> Result<(), ()> {
        let mut q = self.inner.lock().unwrap();
        while q.len() >= self.cap && !self.closed.load(AtomicOrdering::Relaxed) {
            q = self.not_full.wait(q).unwrap();
        }
        if self.closed.load(AtomicOrdering::Relaxed) {
            return Err(());
        }
        let key = g.game.history.len() as u64;
        let seq = g.id;
        q.push(Queued { key, seq, item: g });
        drop(q);
        self.not_empty.notify_one();
        Ok(())
    }

    /// Inference gather. Blocks until `batch` leaves are available, then pops the
    /// `batch` highest-ply games (priority order) under one lock. Returns `None`
    /// once the queue is closed before a full batch is available (shutdown) — the
    /// partial batch is dropped, matching the old disconnect -> discard-partial.
    fn gather(&self, batch: usize) -> Option<Vec<Box<InFlight>>> {
        let mut q = self.inner.lock().unwrap();
        while q.len() < batch && !self.closed.load(AtomicOrdering::Relaxed) {
            q = self.not_empty.wait(q).unwrap();
        }
        if q.len() < batch {
            return None; // closed with only a partial batch left
        }
        let mut games = Vec::with_capacity(batch);
        for _ in 0..batch {
            games.push(q.pop().unwrap().item);
        }
        drop(q);
        self.not_full.notify_all(); // freed up to `batch` slots
        Some(games)
    }

    /// Wake every blocked worker/inference thread and make all further push/gather
    /// fail (shutdown). Replaces the crossbeam channel's drop-based disconnect.
    fn close(&self) {
        self.closed.store(true, AtomicOrdering::Relaxed);
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }
}

pub struct Config {
    pub sims: usize,
    pub add_root_noise: bool,
    pub seed: u64,
    pub n_threads: usize,
    pub n_slots: usize,       // in-flight forwards / pinned input slots
    pub model_path: String,   // serving_model.pt2 (AOTInductor graph, loaded once)
    pub weights_path: String, // serving_weights.safetensors (raw weights swapped in)
    pub reload_every: Duration,
}

struct Shared {
    queue: Mutex<BinaryHeap<Queued>>,
    preinfer: PreInferQueue, // bounded ply-priority queue feeding the inference thread
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
    games_done: AtomicU64,
    plies_done: AtomicU64,
    // --- inference-thread block accounting (where the GPU producer stalls) ---
    infer_queue_wait_us: AtomicU64, // time blocked gathering a full BATCH off the leaf channel (leaf-production bound)
    infer_slot_wait_us: AtomicU64,  // time blocked waiting for a FREE slot (scatter/launch bound)
    infer_forwards: AtomicU64,      // forwards launched
}

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

/// Encode a board and narrow it to bf16 (the dtype the forward consumes), carried
/// as raw bf16 bits in an i16. Run on the worker thread so the f32->bf16 narrowing
/// is parallelized across workers instead of done serially by the single inference
/// thread when it fills a pinned slot.
fn encode_bf16_bits(board: &Board) -> Vec<i16> {
    let mut buf = vec![0.0f32; ENC_LEN];
    board.encode_into(&mut buf);
    buf.iter().map(|&x| bf16::from_f32(x).to_bits() as i16).collect()
}

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

fn apply_result(g: &mut InFlight, res: EvalResult, config: &Config) {
    let ctx = g.leaf_ctx.take().expect("pending result without leaf_ctx");
    // Widen only THIS game's row bf16->f32 here on the worker thread (parallel
    // across workers), instead of widening the whole batch on the scatter thread.
    let logits: Vec<f32> = res.logits[res.row * POLICY_SIZE..(res.row + 1) * POLICY_SIZE]
        .iter()
        .map(|&b| bf16::from_bits(b as u16).to_f32())
        .collect();
    let value = bf16::from_bits(res.value as u16).to_f32();
    match ctx.kind {
        LeafKind::RootExpand => {
            expand_node(&mut g.game.arena, 0, &ctx.board, &logits);
            if config.add_root_noise {
                add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
            }
            g.phase = GamePhase::Simulating(0);
        }
        LeafKind::SimLeaf => {
            let leaf = *ctx.path.last().unwrap();
            expand_node(&mut g.game.arena, leaf, &ctx.board, &logits);
            backprop(&mut g.game.arena, &ctx.path, value as f64);
            if let GamePhase::Simulating(s) = g.phase {
                g.phase = GamePhase::Simulating(s + 1);
            }
        }
    }
}

enum Advance {
    Eval,
    Done(Vec<Row>),
}

fn advance_until_eval_or_done(g: &mut InFlight, config: &Config) -> Advance {
    loop {
        match g.phase {
            GamePhase::NeedRootExpand => {
                if g.game.arena[0].expanded {
                    if config.add_root_noise {
                        add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
                    }
                    g.phase = GamePhase::Simulating(0);
                    continue;
                }
                let board = g.game.board.clone_search();
                g.staged_enc = encode_bf16_bits(&board);
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
                        g.staged_enc = encode_bf16_bits(&board);
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

fn requeue_scattered(shared: &Shared, logits: Arc<Vec<i16>>, values: &[i16], games: Vec<Box<InFlight>>) {
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

/// Worker: pop (or spawn) a game, apply any pending result, advance until it
/// yields one eval (push to the pre-inference queue) or completes (sink), forever
/// until stop. The `push` blocks only when the pre-inference queue is at capacity
/// (backpressure); it returns Err once the queue is closed (shutdown), which ends
/// the loop.
fn worker_loop(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) {
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
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
        if let Some(res) = g.pending.take() {
            apply_result(&mut g, res, &shared.config);
        }
        match advance_until_eval_or_done(&mut g, &shared.config) {
            Advance::Eval => {
                if shared.preinfer.push(g).is_err() {
                    return; // pre-inference queue closed (shutdown)
                }
            }
            Advance::Done(rows) => {
                shared.games_done.fetch_add(1, AtomicOrdering::Relaxed);
                shared.plies_done.fetch_add(rows.len() as u64, AtomicOrdering::Relaxed);
                sink(rows);
            }
        }
    }
}

// --- inference + scatter (overlapped) ---------------------------------------
/// bf16 CUDA tensor -> raw bf16 bits (i16) on host. Just the D2H copy (bf16, half
/// the bytes of an f32 upcast-then-copy); the bf16->f32 widening is deferred to the
/// worker threads (apply_result), which widen only their own row in parallel.
/// tch 0.23 mis-types half::bf16 as Kind::Half, so we bitcast the host bf16 tensor
/// to Int16 (view_dtype = raw reinterpret, no numeric conversion) and copy the bits.
fn bf16_d2h(t: &Tensor) -> Vec<i16> {
    let c = t.to_device(Device::Cpu).contiguous().view_dtype(Kind::Int16);
    let n = c.numel();
    // copy_data writes all `n` elements (and bounds-checks dst.len() >= n) and
    // never reads dst, so copying straight into uninitialized capacity is sound
    // and skips the redundant zero-fill of a buffer we immediately overwrite.
    let mut v: Vec<i16> = Vec::with_capacity(n);
    let dst = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr(), n) };
    c.copy_data(dst, n);
    // copy_data succeeded (it panics on error), so all n elements are initialized.
    unsafe { v.set_len(n) };
    v
}

/// One in-flight forward handed from the inference thread to the scatter thread.
/// Owns its device tensors (Tensor is Send); the scatter thread reads the outputs
/// after `event` and recycles the pinned-input `slot`.
struct WorkItem {
    slot: usize,
    m: usize,
    _x_dev: Tensor, // kept alive until the forward (event) completes
    out_logits: Tensor,
    out_values: Tensor,
    event: CudaEvent,
    games: Vec<Box<InFlight>>,
}

/// Owns the model + a pool of pinned input buffers. The inference thread fills a
/// pinned slot, async-copies it to the device, launches the fused AOTI forward,
/// and records an event — all non-blocking, so forwards queue back-to-back.
struct Infer {
    model: AotiModel,
    dev: Device,
    pinned_in: Vec<Tensor>, // N pinned host bf16 [B,18,8,8]
    weights_path: String,
    reload_every: Duration,
    last_check: Instant,
    last_mtime: Option<std::time::SystemTime>,
}

impl Infer {
    fn new(config: &Config, n_slots: usize) -> Infer {
        let dev = Device::Cuda(0);
        let b = BATCH as i64;
        // The .pt2 defines the forward graph + initial weights and is loaded ONCE;
        // fresh weights arrive via the safetensors sidecar (maybe_reload).
        let model = AotiModel::load(&config.model_path);
        // Guard: the package is exported with a STATIC batch, so it must have been
        // compiled for BATCH. A mismatch either trips the AOTI shape check (caught in
        // the shim -> code 1) or silently yields a wrong-sized output (code 2). Either
        // way, exit the WHOLE process (not just panic this thread) so the worker dies
        // loudly at startup and the orchestrator restarts it, rather than hanging with
        // a dead inference thread and a stalled (silently non-producing) pipeline.
        // This probe also serves as the first warmup forward.
        match model.check_batch(b) {
            0 => {}
            code => {
                eprintln!(
                    "FATAL: {} was not compiled for BATCH={BATCH} (aoti_check_batch={code}); \
                     re-export the .pt2 at this batch (see SERVING_BATCH in training/net.py)",
                    config.model_path
                );
                std::process::exit(1);
            }
        }
        let pinned_in = (0..n_slots)
            .map(|_| Tensor::zeros([b, 18, 8, 8], (Kind::BFloat16, Device::Cpu)).pin_memory(dev))
            .collect();
        Infer {
            model,
            dev,
            pinned_in,
            last_mtime: std::fs::metadata(&config.weights_path).and_then(|m| m.modified()).ok(),
            weights_path: config.weights_path.clone(),
            reload_every: config.reload_every,
            last_check: Instant::now(),
        }
    }

    fn maybe_reload(&mut self) {
        if self.last_check.elapsed() < self.reload_every {
            return;
        }
        self.last_check = Instant::now();
        let m = std::fs::metadata(&self.weights_path).and_then(|md| md.modified()).ok();
        if m.is_none() || m == self.last_mtime {
            return;
        }
        // Trainer published fresh weights: read the safetensors sidecar, move the raw
        // primaries to CUDA bf16, and push them into the live AOTI constant buffer
        // (update_constant_buffer -> run_const_fold -> swap). The trainer writes the
        // file atomically (rename), so a changed mtime means a complete file.
        let entries = match Tensor::read_safetensors(&self.weights_path) {
            Ok(v) => v,
            Err(_) => return, // transient; retry on the next tick
        };
        let mut names: Vec<CString> = Vec::with_capacity(entries.len());
        let mut tensors: Vec<Tensor> = Vec::with_capacity(entries.len());
        for (name, t) in entries {
            // AOTI keys constants by the mangled FQN ('.'->'_').
            names.push(CString::new(name.replace('.', "_")).unwrap());
            tensors.push(t.to_device(self.dev).to_kind(Kind::BFloat16));
        }
        let refs: Vec<&Tensor> = tensors.iter().collect();
        self.model.swap_weights(&names, &refs);
        self.last_mtime = m;
    }

    /// Fill pinned slot `slot` with `m` rows, async-H2D, launch the forward, and
    /// record a completion event. Returns owned device tensors + event (all async
    /// — the CPU does not block on the GPU here). `enc` is already bf16 bits (narrowed
    /// by the workers, carried as i16), so filling the pinned slot is a plain
    /// bf16->bf16 memcpy with no per-element conversion on this single thread.
    fn launch(&mut self, slot: usize, enc: &[i16], m: usize) -> (Tensor, Tensor, Tensor, CudaEvent) {
        // i16 bits -> bf16 view (bitcast, no numeric conversion), then memcpy into pinned.
        let src = Tensor::from_slice(enc).reshape([m as i64, 18, 8, 8]).view_dtype(Kind::BFloat16);
        self.pinned_in[slot].narrow(0, 0, m as i64).copy_(&src); // bf16 -> bf16 (memcpy) into pinned
        // async H2D from pinned memory (does not block the CPU).
        let x = self.pinned_in[slot].internal_to_copy((Kind::BFloat16, self.dev), true);
        let opts = (Kind::BFloat16, self.dev);
        let b = self.pinned_in[slot].size()[0];
        let out_logits = Tensor::zeros([b, POLICY_SIZE as i64], opts);
        let out_values = Tensor::zeros([b], opts);
        self.model.run(&x, &out_logits, &out_values); // async on the current stream
        let event = CudaEvent::new();
        event.record();
        (x, out_logits, out_values, event)
    }
}

/// Producer: gather BATCH leaves off the channel into a free pinned slot, launch
/// the forward, and hand the in-flight slot to the scatter thread. Blocks while
/// gathering a full batch (leaf-production bound) or waiting for a free slot
/// (scatter/launch bound).
fn inference_thread(
    shared: Arc<Shared>,
    work_tx: SyncSender<WorkItem>,
    free_rx: Receiver<usize>,
) {
    cudagraph::set_side_stream(); // a non-default stream for all inference ops
    let mut infer = Infer::new(&shared.config, shared.config.n_slots.max(1));
    let mut enc: Vec<i16> = Vec::with_capacity(BATCH * ENC_LEN); // reused scratch (bf16 bits as i16, staged by workers)
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        // Gather the BATCH highest-ply leaves off the pre-inference queue (priority
        // order). gather blocks until a full BATCH is available; None means the
        // queue is closed with only a partial batch (shutdown) -> drop it and exit.
        let t_queue = Instant::now();
        let games = match shared.preinfer.gather(BATCH) {
            Some(g) => g,
            None => return,
        };
        shared.infer_queue_wait_us.fetch_add(t_queue.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
        enc.clear();
        for g in &games {
            enc.extend_from_slice(&g.staged_enc);
        }
        infer.maybe_reload(); // excluded from both waits (occasional file I/O, not a stall on the pipeline)
        let t_slot = Instant::now();
        let slot = match free_rx.recv() {
            Ok(s) => s,
            Err(_) => return, // scatter gone
        };
        shared.infer_slot_wait_us.fetch_add(t_slot.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        shared.infer_forwards.fetch_add(1, AtomicOrdering::Relaxed);
        let m = games.len(); // == BATCH
        let (x, out_logits, out_values, event) = infer.launch(slot, &enc, m);
        let item = WorkItem { slot, m, _x_dev: x, out_logits, out_values, event, games };
        if work_tx.send(item).is_err() {
            return;
        }
    }
}

/// Consumer: wait each forward's event, read its bf16 outputs (D2H), requeue the
/// games, and recycle the pinned slot. Overlaps the inference thread's next forward.
fn scatter_thread(shared: Arc<Shared>, work_rx: Receiver<WorkItem>, free_tx: SyncSender<usize>) {
    while let Ok(item) = work_rx.recv() {
        item.event.sync(); // wait the forward + output copy
        let m = item.m as i64;
        let logits = Arc::new(bf16_d2h(&item.out_logits.narrow(0, 0, m)));
        let values = bf16_d2h(&item.out_values.narrow(0, 0, m));
        requeue_scattered(&shared, logits, &values, item.games);
        if free_tx.send(item.slot).is_err() {
            return;
        }
        // item's device tensors + event drop here (after the readback).
    }
}

fn make_shared(config: Config) -> Arc<Shared> {
    // Pre-inference queue cap: 2 * (n_slots * slot_size), slot_size == BATCH.
    let preinfer_cap = 2 * config.n_slots.max(1) * BATCH;
    Arc::new(Shared {
        queue: Mutex::new(BinaryHeap::new()),
        preinfer: PreInferQueue::new(preinfer_cap),
        stop: AtomicBool::new(false),
        id_counter: AtomicU64::new(0),
        games_done: AtomicU64::new(0),
        plies_done: AtomicU64::new(0),
        infer_queue_wait_us: AtomicU64::new(0),
        infer_slot_wait_us: AtomicU64::new(0),
        infer_forwards: AtomicU64::new(0),
        config,
    })
}

struct Pipeline {
    workers: Vec<thread::JoinHandle<()>>,
    infer: thread::JoinHandle<()>,
    scatter: thread::JoinHandle<()>,
}

fn spawn_pipeline(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) -> Pipeline {
    let n_slots = shared.config.n_slots.max(1);
    // Workers push leaves into the bounded ply-priority pre-inference queue
    // (shared.preinfer); the inference thread pops the highest-ply BATCH. Shutdown
    // is driven by shared.preinfer.close() (see `shutdown`), which wakes everyone
    // blocked in push/gather — replacing the old channel's drop-based disconnect.
    let n_threads = shared.config.n_threads.max(1);
    let mut workers = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let sh = shared.clone();
        let sk = sink.clone();
        workers.push(thread::spawn(move || worker_loop(sh, sk)));
    }

    let (work_tx, work_rx) = sync_channel::<WorkItem>(n_slots);
    let (free_tx, free_rx) = sync_channel::<usize>(n_slots);
    for i in 0..n_slots {
        free_tx.send(i).unwrap();
    }
    let infer = {
        let sh = shared.clone();
        thread::spawn(move || inference_thread(sh, work_tx, free_rx))
    };
    let scatter = {
        let sh = shared.clone();
        thread::spawn(move || scatter_thread(sh, work_rx, free_tx))
    };
    Pipeline { workers, infer, scatter }
}

fn shutdown(shared: &Shared, p: Pipeline) {
    shared.stop.store(true, AtomicOrdering::Relaxed);
    shared.preinfer.close(); // wake everyone parked in push/gather, fail further ops
    // Workers see `stop` at the loop top (or get Err from a parked push) and exit.
    // The inference thread returns at its stop-check or when gather sees the closed
    // queue, and drops work_tx, ending the scatter thread once its queue drains.
    // Joining workers first is safe: scatter keeps freeing slots, so the inference
    // thread never wedges waiting on a slot during teardown.
    for h in p.workers {
        let _ = h.join();
    }
    let _ = p.infer.join();
    let _ = p.scatter.join();
}

/// Bench mode: run for `run` seconds, printing games/sec + avg_plies + rows/sec
/// every `interval` seconds (mirrors bench_games_sec.py). Exclusive GPU expected.
pub fn run_bench(config: Config, run: Duration, interval: Duration) {
    let shared = make_shared(config);
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(|_rows: Vec<Row>| {});
    let pipeline = spawn_pipeline(shared.clone(), sink);

    let start = Instant::now();
    let mut win_start = start;
    let (mut last_games, mut last_plies) = (0u64, 0u64);
    let (mut last_bw, mut last_sw, mut last_fwd) = (0u64, 0u64, 0u64);
    println!("selfplay_rs bench: batch={} threads={} slots={} sims={}",
             BATCH, shared.config.n_threads, shared.config.n_slots, shared.config.sims);
    while start.elapsed() < run {
        thread::sleep(Duration::from_millis(200));
        if win_start.elapsed() >= interval {
            let dt = win_start.elapsed().as_secs_f64();
            let g = shared.games_done.load(AtomicOrdering::Relaxed);
            let p = shared.plies_done.load(AtomicOrdering::Relaxed);
            let (dg, dp) = (g - last_games, p - last_plies);
            // inference-thread block accounting over the window: % of wall time the
            // single GPU-producer thread was blocked gathering a full batch off the
            // leaf channel (leaf-production bound) vs. a free slot (scatter/launch bound).
            let bw = shared.infer_queue_wait_us.load(AtomicOrdering::Relaxed);
            let sw = shared.infer_slot_wait_us.load(AtomicOrdering::Relaxed);
            let fwd = shared.infer_forwards.load(AtomicOrdering::Relaxed);
            let (dbw, dsw, dfwd) = (bw - last_bw, sw - last_sw, fwd - last_fwd);
            let dt_us = dt * 1e6;
            println!(
                "[+{:5.0}s] {:6.3} games/sec  ({:5.1} avg plies, {:5.0} rows/sec)  | infer: queue_wait {:4.1}%  slot_wait {:4.1}%  ({:5.0} fwd/s)",
                start.elapsed().as_secs_f64(),
                dg as f64 / dt,
                dp as f64 / dg.max(1) as f64,
                dp as f64 / dt,
                dbw as f64 / dt_us * 100.0,
                dsw as f64 / dt_us * 100.0,
                dfwd as f64 / dt,
            );
            last_games = g;
            last_plies = p;
            last_bw = bw;
            last_sw = sw;
            last_fwd = fwd;
            win_start = Instant::now();
        }
    }
    shutdown(&shared, pipeline);
    println!("DONE (read steady-state from the last few intervals)");
}

// --- serve mode (production IPC) ---------------------------------------------
/// Write `&[f32]` as little-endian bytes (aarch64/x86 are LE; matches numpy
/// frombuffer('<f4') on the reader side). Contiguous slice -> raw bytes.
fn write_f32s<W: Write>(w: &mut W, s: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) };
    w.write_all(bytes)
}

/// One finished game = one frame: u32 n_rows, then n_rows x (state[1152] f32,
/// pi[4672] f32, z f32), all little-endian. Flushed per game so the trainer's
/// reader sees data promptly.
fn write_frame(out: &Mutex<BufWriter<io::Stdout>>, rows: &[Row]) {
    let mut w = out.lock().unwrap();
    let _ = w.write_all(&(rows.len() as u32).to_le_bytes());
    for (state, pi, z) in rows {
        let _ = write_f32s(&mut *w, state);
        let _ = write_f32s(&mut *w, pi);
        let _ = w.write_all(&z.to_le_bytes());
    }
    let _ = w.flush();
}

/// Serve mode: run the pipeline forever, streaming finished games as framed bytes
/// on stdout for the orchestrator's reader thread. Stops when the parent closes
/// our stdin (EOF = orchestrator died) or on SIGTERM (default-kills the process).
pub fn run_serve(config: Config) {
    let shared = make_shared(config);
    let stdout = Arc::new(Mutex::new(BufWriter::new(io::stdout())));
    let so = stdout.clone();
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(move |rows: Vec<Row>| write_frame(&so, &rows));
    let pipeline = spawn_pipeline(shared.clone(), sink);

    // Block until the orchestrator closes our stdin (its death closes the pipe).
    let mut buf = [0u8; 256];
    loop {
        match io::stdin().read(&mut buf) {
            Ok(0) | Err(_) => break, // EOF / error -> shut down
            Ok(_) => {}              // ignore any bytes
        }
    }
    shutdown(&shared, pipeline);
    let _ = stdout.lock().unwrap().flush();
}
