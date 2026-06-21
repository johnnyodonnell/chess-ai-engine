//! Decoupled multi-threaded self-play pipeline (ported from the pattern in
//! fox-lite `selfplay_rs/src/pipeline.rs` and this repo's AlphaZero worker).
//!
//! Self-play is CPU-bound: legal-move generation + building the 4672-wide legal
//! mask + sampling dominate, while the tiny ChessNet GPU forward is ~1-2ms and
//! the GPU otherwise sits idle. So we split the work:
//!   - N **worker threads** run game logic on the CPU (sample the returned
//!     policy, apply the move, advance to the next decision — encoding it and
//!     building its legal mask — or finalize the game and emit its reward rows).
//!   - one **inference thread** gathers a batch of pending decisions, runs a
//!     single GPU forward, and routes each result back to its owning worker.
//! With `concurrency` (>= batch) games in flight, the inference thread always has
//! a batch ready while workers chew on the previous results — CPU game logic
//! overlaps the GPU forward, and the CPU work fans out across all cores.
//!
//! Each game stays with the worker that owns it (sticky), so the return path is a
//! per-worker single-consumer channel (std mpsc); the submit path is one shared
//! multi-producer channel into the inference thread. Cohort format / reward are
//! identical to the single-thread `selfplay::run`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use chess_core::{pack_move, Board, ENC_LEN};

use crate::net::Net;
use crate::selfplay::{pick_device, ply_temp, sample_action, Config, POLICY_SIZE, ROW_FLOATS};

struct Decision {
    state: Vec<f32>,
    mask: Vec<f32>,
    action: u32,
    white_to_move: bool,
}

/// Everything needed to sample + apply the current decision, computed by the
/// worker *before* the forward (none of it depends on the logits).
struct Pending {
    enc: Vec<f32>,         // encoded current position (the net input == decision state)
    cand: Vec<(usize, u16)>, // (policy_index, packed_move) per legal move
    mask: Vec<f32>,        // 4672 legal mask
    white_to_move: bool,
}

struct Game {
    board: Board,
    decisions: Vec<Decision>,
    plies: u32,
}

struct InFlight {
    game: Game,
    pending: Pending,
    owner: usize,
}

struct Shared {
    rows_emitted: AtomicUsize,
    in_flight: AtomicUsize,
    done: AtomicBool,
    target: usize,
    max_plies: u32,
    temperature: f64,
    temp_end: f64,
}

fn new_game(board: Board) -> Game {
    Game { board, decisions: Vec::new(), plies: 0 }
}

/// Encode the position and build the legal candidate list + mask. Only called on
/// non-terminal boards (a terminal board has no legal moves).
fn prepare(board: &Board) -> Pending {
    let mut enc = vec![0f32; ENC_LEN];
    board.encode_into(&mut enc);
    let mut mask = vec![0f32; POLICY_SIZE];
    let mut cand = Vec::with_capacity(48);
    for mv in board.legal_moves_vec() {
        let idx = board.move_to_index(mv) as usize;
        mask[idx] = 1.0;
        cand.push((idx, pack_move(mv)));
    }
    Pending { enc, cand, mask, white_to_move: board.turn() }
}

/// Drain a finished game's decisions into the worker's local row buffer, scoring
/// each by the terminal outcome from that mover's POV. Returns rows emitted.
fn finalize(game: &mut Game, buf: &mut Vec<f32>) -> usize {
    let ow = game.board.outcome_white().unwrap_or(0) as f32; // ply cap => draw 0
    let mut k = 0;
    for d in game.decisions.drain(..) {
        let z = if d.white_to_move { ow } else { -ow };
        buf.extend_from_slice(&d.state);
        buf.extend_from_slice(&d.mask);
        buf.push(d.action as f32);
        buf.push(z);
        k += 1;
    }
    k
}

/// One forward's worth of logits, shared by all games in that batch (avoids a
/// per-game 4672-wide copy in the inference thread). `(batch_logits, my_index)`.
type Reply = (InFlight, Arc<Vec<f32>>, usize);

/// Worker thread: own games, sample + apply moves on returned logits, advance or
/// finalize. Returns (row buffer, rows) for the cohort merge.
fn worker(
    id: usize,
    rx: Receiver<Reply>,
    to_infer: Sender<InFlight>,
    shared: Arc<Shared>,
    mut rng: StdRng,
) -> (Vec<f32>, usize) {
    let mut buf: Vec<f32> = Vec::new();
    let mut rows = 0usize;

    while let Ok((mut inf, logits, j)) = rx.recv() {
        let row = &logits[j * POLICY_SIZE..(j + 1) * POLICY_SIZE];
        let temp = ply_temp(shared.temperature, shared.temp_end, inf.game.plies);
        let k = sample_action(&inf.pending.cand, row, temp, &mut rng);
        let (action_idx, handle) = inf.pending.cand[k]; // (usize,u16): Copy

        let Pending { enc, mask, white_to_move, .. } = inf.pending;
        inf.game.decisions.push(Decision { state: enc, mask, action: action_idx as u32, white_to_move });
        inf.game.board.push(handle);
        inf.game.plies += 1;

        let terminal =
            inf.game.board.outcome_white().is_some() || inf.game.plies >= shared.max_plies;
        if terminal {
            let k = finalize(&mut inf.game, &mut buf);
            rows += k;
            let total = shared.rows_emitted.fetch_add(k, Ordering::Relaxed) + k;
            if total < shared.target {
                // Replace the finished game (in_flight unchanged).
                let board = Board::new();
                let pending = prepare(&board);
                if to_infer.send(InFlight { game: new_game(board), pending, owner: id }).is_err() {
                    break;
                }
            } else if shared.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
                shared.done.store(true, Ordering::Release); // last game retired
            }
        } else {
            inf.pending = prepare(&inf.game.board);
            if to_infer.send(inf).is_err() {
                break;
            }
        }
    }
    (buf, rows)
}

/// Inference thread: gather a batch of pending decisions, one GPU forward, route
/// each result back to its owning worker. Exits once `done` is set and the
/// pending queue drains; dropping the worker senders then unblocks the workers.
fn inference(
    net: Net,
    dev: Device,
    rx: Receiver<InFlight>,
    workers: Vec<Sender<Reply>>,
    shared: Arc<Shared>,
    batch: usize,
) {
    loop {
        let first = match rx.recv_timeout(Duration::from_millis(5)) {
            Ok(x) => x,
            Err(RecvTimeoutError::Timeout) => {
                if shared.done.load(Ordering::Acquire) {
                    break;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut items = Vec::with_capacity(batch);
        items.push(first);
        while items.len() < batch {
            match rx.try_recv() {
                Ok(x) => items.push(x),
                Err(_) => break,
            }
        }

        let m = items.len();
        let mut enc_flat = Vec::with_capacity(m * ENC_LEN);
        for inf in &items {
            enc_flat.extend_from_slice(&inf.pending.enc);
        }
        let x = Tensor::from_slice(&enc_flat).reshape([m as i64, 18, 8, 8]).to_device(dev);
        let (logits, _) = tch::no_grad(|| net.forward(&x));
        let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu).contiguous();
        let mut lbuf = vec![0f32; m * POLICY_SIZE];
        logits.copy_data(&mut lbuf, m * POLICY_SIZE);
        let shared_logits = Arc::new(lbuf);

        for (j, inf) in items.into_iter().enumerate() {
            let owner = inf.owner;
            let _ = workers[owner].send((inf, shared_logits.clone(), j));
        }
    }
    // `workers` (the result senders) drop here, unblocking each worker's recv.
}

/// Run one cohort with the pipeline; returns rows written.
pub fn run(cfg: &Config) -> usize {
    run_on(cfg, pick_device(cfg.cpu))
}

pub fn run_on(cfg: &Config, dev: Device) -> usize {
    // The forward runs on the GPU; libtorch's default CPU intra-op pool (one
    // thread per core) would otherwise fight our game-worker threads and cause
    // oversubscription slowdowns. Pin it to 1 so the cores go to the workers.
    tch::set_num_threads(1);

    let net = Net::load(&cfg.weights, dev, Kind::Float);
    let n_threads = cfg.threads.max(1);
    let batch = cfg.batch.max(1);
    // 0 => 2*batch: keep ~a batch of games being processed by workers while the
    // inference thread runs the forward on the other batch (double-buffer).
    let concurrency = if cfg.concurrency == 0 { 2 * batch } else { cfg.concurrency.max(batch) };

    let shared = Arc::new(Shared {
        rows_emitted: AtomicUsize::new(0),
        in_flight: AtomicUsize::new(concurrency),
        done: AtomicBool::new(false),
        target: cfg.target_rows,
        max_plies: cfg.max_plies,
        temperature: cfg.temperature,
        temp_end: cfg.temp_end,
    });

    let (to_infer_tx, to_infer_rx) = mpsc::channel::<InFlight>();
    let mut worker_txs: Vec<Sender<Reply>> = Vec::with_capacity(n_threads);
    let mut worker_rxs = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let (t, r) = mpsc::channel();
        worker_txs.push(t);
        worker_rxs.push(r);
    }

    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_threads);
    for (i, rx_i) in worker_rxs.into_iter().enumerate() {
        let to_infer = to_infer_tx.clone();
        let sh = shared.clone();
        let seed = cfg.seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        handles.push(thread::spawn(move || worker(i, rx_i, to_infer, sh, StdRng::seed_from_u64(seed))));
    }

    // Seed the in-flight games round-robin across workers (the channel is
    // unbounded, so this buffers fine before the inference thread starts).
    for j in 0..concurrency {
        let board = Board::new();
        let pending = prepare(&board);
        to_infer_tx
            .send(InFlight { game: new_game(board), pending, owner: j % n_threads })
            .unwrap();
    }
    drop(to_infer_tx); // now only workers hold submit senders

    let inf_handle = {
        let sh = shared.clone();
        thread::spawn(move || inference(net, dev, to_infer_rx, worker_txs, sh, batch))
    };

    let mut buffers: Vec<Vec<f32>> = Vec::with_capacity(n_threads);
    let mut total = 0usize;
    for h in handles {
        let (buf, rows) = h.join().expect("worker panicked");
        total += rows;
        buffers.push(buf);
    }
    inf_handle.join().expect("inference panicked");

    write_cohort(&cfg.out, &buffers, total);

    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "self-play(pipeline t={n_threads} c={concurrency} b={batch}): {total} rows, {:.1}s, {:.0} rows/s",
        secs,
        total as f64 / secs,
    );
    total
}

fn write_cohort(path: &str, buffers: &[Vec<f32>], n_rows: usize) {
    let f = File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    let mut w = BufWriter::new(f);
    w.write_all(&(n_rows as u32).to_le_bytes()).unwrap();
    w.write_all(&(ROW_FLOATS as u32).to_le_bytes()).unwrap();
    for b in buffers {
        // f32 slice -> &[u8] (little-endian on x86/ARM; matches read_cohort).
        let bytes = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4) };
        w.write_all(bytes).unwrap();
    }
    w.flush().unwrap();
}
