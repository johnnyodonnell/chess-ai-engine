//! Shared self-play config + helpers (cohort format, sampling, temperature,
//! device pick). The actual game generation runs in `pipeline` (multi-threaded,
//! GPU-overlapped); this module holds the pieces both it and the CLI share.
//!
//! Cohort row layout (ROW_FLOATS = ENC_LEN + POLICY_SIZE + 2):
//!   [ state(1152) | legal_mask(4672) | action_index | z ]
//! File: u32 num_rows (LE), u32 row_floats (LE), then num_rows*row_floats f32 LE.
//!
//! Reward is vanilla REINFORCE: each decision is scored by the terminal game
//! outcome from that mover's POV (z = +1 win / 0 draw / -1 loss, undiscounted),
//! which lines up with ChessNet's side-to-move value head.

use rand::rngs::StdRng;
use rand::Rng;
use tch::Device;

use chess_core::ENC_LEN;

/// AlphaZero policy size: 64 squares * 73 planes.
pub const POLICY_SIZE: usize = 4672;
pub const ROW_FLOATS: usize = ENC_LEN + POLICY_SIZE + 2;

pub struct Config {
    pub weights: String,
    pub out: String,
    /// Collect at least this many decision rows, then let in-flight games finish.
    pub target_rows: usize,
    /// Max GPU forward width (positions per batched forward).
    pub batch: usize,
    /// Number of CPU game-worker threads.
    pub threads: usize,
    /// Games kept in flight (>= batch; overlaps CPU game logic with the GPU
    /// forward). 0 => 2*batch.
    pub concurrency: usize,
    pub temperature: f64,
    pub temp_end: f64,
    /// Hard ply cap; a game reaching it is scored as a draw (z = 0).
    pub max_plies: u32,
    pub seed: u64,
    pub cpu: bool,
}

pub fn pick_device(cpu: bool) -> Device {
    if cpu {
        Device::Cpu
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        eprintln!("warning: CUDA unavailable, self-play on CPU");
        Device::Cpu
    }
}

/// Per-ply sampling temperature: hold `start` for the opening, then anneal
/// linearly to `end` (a slightly noisy opening for diversity, sharper later).
pub(crate) fn ply_temp(start: f64, end: f64, ply: u32) -> f64 {
    const OPENING: f64 = 20.0;
    const ANNEAL: f64 = 40.0;
    let p = ply as f64;
    if p < OPENING {
        return start;
    }
    let frac = ((p - OPENING) / ANNEAL).clamp(0.0, 1.0);
    start + (end - start) * frac
}

/// Temperature softmax over legal logits; returns the sampled candidate's index
/// into `cand` (each entry is (policy_index, packed_move_handle)).
pub(crate) fn sample_action(cand: &[(usize, u16)], logits: &[f32], temp: f64, rng: &mut StdRng) -> usize {
    let t = temp.max(1e-6) as f32;
    let maxl = cand
        .iter()
        .map(|&(idx, _)| logits[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = cand.iter().map(|&(idx, _)| ((logits[idx] - maxl) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let r = rng.gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (k, _) in cand.iter().enumerate() {
        acc += exps[k];
        if r <= acc {
            return k;
        }
    }
    cand.len() - 1
}
