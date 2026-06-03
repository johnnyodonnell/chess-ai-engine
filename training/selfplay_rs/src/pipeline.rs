//! Decoupled non-blocking self-play pipeline, 100% Rust (no GIL, no PyO3).
//!
//! Port of the Rust half of the old `chess_rs/src/pipeline.rs` with the entire
//! Python-boundary apparatus removed: the consumer/scatter split, the
//! `Arc<Vec<f32>>` copy-out under the GIL, the numpy round-trip. Here ONE
//! inference thread owns the tch net + a CUDA-graphed static device slot and
//! talks to the worker threads with plain Rust data.
//!
//! Topology (mirrors the original):
//!   - global ply-priority queue of in-flight games (most plies first);
//!   - N worker threads advance a game's FSM until it yields ONE leaf eval (push
//!     to the active bucket) or completes (hand rows to the sink); empty queue =>
//!     spawn a fresh game (dynamic game count, self-regulates to ~2*bucket);
//!   - two buckets: workers fill one while the inference thread runs the other;
//!   - ONE inference thread: copies a full bucket into the static bf16 device
//!     input, replays the captured CUDA graph (net forward + output copy = one
//!     launch), reads bf16 outputs back, and requeues the games with results.
//!
//! Weight hot-reload: the inference thread mtime-polls the safetensors sidecar
//! and reloads IN PLACE (Net::reload), so the captured graph keeps replaying the
//! fresh weights with no recapture.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chess_core::board::Board;
use chess_core::mcts::{
    add_dirichlet_noise, backprop, copy_subtree, expand_node, sample_move, visits_to_pi,
    walk_to_leaf, Game, Node, ENC_LEN, MAX_PLIES, POLICY_SIZE,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use crate::aoti::AotiModel;

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
    logits: Arc<Vec<f32>>, // full batch policy [m * POLICY_SIZE]
    row: usize,
    value: f32,
}
struct InFlight {
    id: u64,
    game: Game,
    rng: StdRng,
    phase: GamePhase,
    pending: Option<EvalResult>,
    leaf_ctx: Option<LeafCtx>,
    staged_enc: Vec<f32>,
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

struct Bucket {
    enc: Vec<f32>,
    games: Vec<Box<InFlight>>,
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
    active: Bucket,
    ready: Option<Bucket>,
}

pub struct Config {
    pub sims: usize,
    pub add_root_noise: bool,
    pub bucket_size: usize,
    pub seed: u64,
    pub n_threads: usize,
    pub model_path: String, // serving_model.pt2 (AOTInductor package)
    pub reload_every: Duration,
}

struct Shared {
    queue: Mutex<BinaryHeap<Queued>>,
    buckets: Mutex<BucketState>,
    cv_worker: Condvar,
    cv_infer: Condvar,
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
    games_done: AtomicU64,
    plies_done: AtomicU64,
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
            return;
        }
        if bs.active.len() < b {
            bs.active.enc.extend_from_slice(&g.staged_enc);
            bs.active.games.push(g);
            if bs.active.len() == b && bs.ready.is_none() {
                let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
                bs.ready = Some(full);
                shared.cv_infer.notify_one();
            }
            return;
        }
        if bs.ready.is_none() {
            let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
            bs.ready = Some(full);
            shared.cv_infer.notify_one();
            continue;
        }
        bs = shared.cv_worker.wait(bs).unwrap();
    }
}

fn requeue_scattered(shared: &Shared, logits: Arc<Vec<f32>>, values: &[f32], games: Vec<Box<InFlight>>) {
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
/// yields one eval (push to bucket) or completes (sink), forever until stop.
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
            Advance::Eval => push_to_bucket(&shared, g),
            Advance::Done(rows) => {
                shared.games_done.fetch_add(1, AtomicOrdering::Relaxed);
                shared.plies_done.fetch_add(rows.len() as u64, AtomicOrdering::Relaxed);
                sink(rows);
            }
        }
    }
}

// --- inference thread --------------------------------------------------------
/// bf16 CUDA tensor -> Vec<f32> on host. D2H is bf16 (half the bytes of an f32
/// upcast-then-copy); the bf16->f32 widening happens on the CPU.
fn bf16_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let c = t.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let n = c.numel();
    let mut v = vec![0f32; n];
    c.copy_data(&mut v, n);
    v
}

struct Infer {
    model: AotiModel,
    dev: Device,
    b: usize,
    in_dev: Tensor,     // [B,18,8,8] bf16 (static input the .pt2 reads)
    out_logits: Tensor, // [B,4672] bf16 (AOTI output target)
    out_values: Tensor, // [B] bf16
    model_path: String, // serving_model.pt2
    reload_every: Duration,
    last_check: Instant,
    last_mtime: Option<std::time::SystemTime>,
}

impl Infer {
    fn new(config: &Config) -> Infer {
        let dev = Device::Cuda(0);
        let b = config.bucket_size;
        let model = AotiModel::load(&config.model_path);
        let opts = (Kind::BFloat16, dev);
        Infer {
            model,
            dev,
            b,
            in_dev: Tensor::zeros([b as i64, 18, 8, 8], opts),
            out_logits: Tensor::zeros([b as i64, POLICY_SIZE as i64], opts),
            out_values: Tensor::zeros([b as i64], opts),
            last_mtime: std::fs::metadata(&config.model_path).and_then(|m| m.modified()).ok(),
            model_path: config.model_path.clone(),
            reload_every: config.reload_every,
            last_check: Instant::now(),
        }
    }

    fn maybe_reload(&mut self) {
        if self.last_check.elapsed() < self.reload_every {
            return;
        }
        self.last_check = Instant::now();
        let m = std::fs::metadata(&self.model_path).and_then(|md| md.modified()).ok();
        if m.is_some() && m != self.last_mtime {
            // Trainer recompiled the .pt2 with fresh weights: drop the old loader,
            // load the new package (~ms). No graph to recapture.
            self.model = AotiModel::load(&self.model_path);
            self.last_mtime = m;
        }
    }

    /// Run the fused AOTI forward on a bucket of `m` encoded rows; return the full
    /// policy [m*POLICY_SIZE] + values [m]. The .pt2 is compiled for fixed B; a
    /// partial batch (startup/shutdown) fills the first m rows and ignores the rest.
    fn infer(&mut self, enc: &[f32], m: usize) -> (Arc<Vec<f32>>, Vec<f32>) {
        self.maybe_reload();
        let host = Tensor::from_slice(enc).reshape([m as i64, 18, 8, 8]);
        let x = host.to_device(self.dev).to_kind(Kind::BFloat16);
        // Fill the first m rows of the static input (the .pt2 always runs B rows).
        self.in_dev.narrow(0, 0, m as i64).copy_(&x);
        self.model.run(&self.in_dev, &self.out_logits, &self.out_values);
        let logits = Arc::new(bf16_to_vec_f32(&self.out_logits.narrow(0, 0, m as i64)));
        let values = bf16_to_vec_f32(&self.out_values.narrow(0, 0, m as i64));
        (logits, values)
    }
}

fn inference_loop(shared: Arc<Shared>) {
    let mut infer = Infer::new(&shared.config);
    loop {
        // acquire a full bucket, polling stop while idle.
        let bucket = loop {
            let mut bs = shared.buckets.lock().unwrap();
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return;
            }
            if let Some(b) = bs.ready.take() {
                shared.cv_worker.notify_all();
                break b;
            }
            let _ = shared.cv_infer.wait_timeout(bs, Duration::from_millis(200)).unwrap();
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return;
            }
        };
        let Bucket { enc, games } = bucket;
        let m = games.len();
        let (logits, values) = infer.infer(&enc, m);
        requeue_scattered(&shared, logits, &values, games);
    }
}

fn make_shared(config: Config) -> Arc<Shared> {
    let bucket_size = config.bucket_size.max(1);
    Arc::new(Shared {
        queue: Mutex::new(BinaryHeap::new()),
        buckets: Mutex::new(BucketState {
            active: Bucket::with_capacity(bucket_size),
            ready: None,
        }),
        cv_worker: Condvar::new(),
        cv_infer: Condvar::new(),
        stop: AtomicBool::new(false),
        id_counter: AtomicU64::new(0),
        games_done: AtomicU64::new(0),
        plies_done: AtomicU64::new(0),
        config,
    })
}

fn spawn_pipeline(
    shared: Arc<Shared>,
    sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>,
) -> (Vec<thread::JoinHandle<()>>, thread::JoinHandle<()>) {
    let n_threads = shared.config.n_threads.max(1);
    let mut workers = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let sh = shared.clone();
        let sk = sink.clone();
        workers.push(thread::spawn(move || worker_loop(sh, sk)));
    }
    let sh = shared.clone();
    let infer = thread::spawn(move || inference_loop(sh));
    (workers, infer)
}

fn shutdown(shared: &Shared, workers: Vec<thread::JoinHandle<()>>, infer: thread::JoinHandle<()>) {
    shared.stop.store(true, AtomicOrdering::Relaxed);
    shared.cv_worker.notify_all();
    shared.cv_infer.notify_all();
    for h in workers {
        let _ = h.join();
    }
    let _ = infer.join();
}

/// Bench mode: run for `run` seconds, printing games/sec + avg_plies + rows/sec
/// every `interval` seconds (mirrors bench_games_sec.py). Exclusive GPU expected.
pub fn run_bench(config: Config, run: Duration, interval: Duration) {
    let shared = make_shared(config);
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(|_rows: Vec<Row>| {});
    let (workers, infer) = spawn_pipeline(shared.clone(), sink);

    let start = Instant::now();
    let mut win_start = start;
    let (mut last_games, mut last_plies) = (0u64, 0u64);
    println!("selfplay_rs bench: bucket={} threads={} sims={}", shared.config.bucket_size, shared.config.n_threads, shared.config.sims);
    while start.elapsed() < run {
        thread::sleep(Duration::from_millis(200));
        if win_start.elapsed() >= interval {
            let dt = win_start.elapsed().as_secs_f64();
            let g = shared.games_done.load(AtomicOrdering::Relaxed);
            let p = shared.plies_done.load(AtomicOrdering::Relaxed);
            let (dg, dp) = (g - last_games, p - last_plies);
            println!(
                "[+{:5.0}s] {:6.3} games/sec  ({:5.1} avg plies, {:5.0} rows/sec)",
                start.elapsed().as_secs_f64(),
                dg as f64 / dt,
                dp as f64 / dg.max(1) as f64,
                dp as f64 / dt,
            );
            last_games = g;
            last_plies = p;
            win_start = Instant::now();
        }
    }
    shutdown(&shared, workers, infer);
    println!("DONE (read steady-state from the last few intervals)");
}
