//! evaluate_reinforce — pool-Elo evaluation for the REINFORCE chess trainer.
//!
//! Ported from this repo's AlphaZero `selfplay_rs/src/eval.rs`, but the players
//! are **greedy raw-policy** (a single batched forward, argmax over legal moves —
//! the way the REINFORCE net actually plays) rather than MCTS. The opening plies
//! are temperature-sampled for game diversity (otherwise greedy-vs-greedy replays
//! one game). Mechanics otherwise match eval.rs / fox-lite's evaluate_rs:
//!
//! A candidate snapshot plays match games against an active pool chosen by rating
//! (top-`n_top` + a `random` floor anchor pinned at Elo 0 + `n_anchors` frozen
//! snapshots spread across the Elo range). A global Bradley-Terry Elo (draws
//! count 0.5) is refit over all accumulated results, and pool.json is updated.
//! Promotion to the browser model stays manual (no auto-serve).
//!
//!   evaluate_reinforce --run-dir runs/run1 --candidate <snap>.safetensors
//!       [--games 60] [--n-top 2] [--n-anchors 3] [--opening-plies 8]
//!       [--max-plies 200] [--seed 0]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use chess_core::{pack_move, Board, ENC_LEN};
use indexmap::IndexMap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use selfplay_reinforce::net::Net;
use selfplay_reinforce::selfplay::POLICY_SIZE;

const RANDOM: &str = "random"; // reserved name for the fixed floor anchor (Elo 0)
const OPENING_TEMP: f64 = 1.0; // temperature for opening plies (game diversity)

fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

// ---------------------------------------------------------------------------
// Players (greedy raw-policy; opening plies temperature-sampled for diversity)
// ---------------------------------------------------------------------------
trait Player {
    /// Pick a packed move for each board (the group where this player is to
    /// move). `plies[i]` is board `i`'s ply, gating opening-temp sampling.
    fn select_moves(&self, boards: &[&Board], plies: &[usize], opening_plies: usize,
                    rng: &mut StdRng) -> Vec<u16>;
}

struct RandomPlayer;

impl Player for RandomPlayer {
    fn select_moves(&self, boards: &[&Board], _plies: &[usize], _opening: usize,
                    rng: &mut StdRng) -> Vec<u16> {
        boards
            .iter()
            .map(|b| {
                let moves = b.legal_moves_vec();
                pack_move(moves[rng.gen_range(0..moves.len())])
            })
            .collect()
    }
}

struct NetPlayer {
    net: Net,
    dev: Device,
}

impl NetPlayer {
    fn load(st_path: &Path, dev: Device) -> NetPlayer {
        NetPlayer { net: Net::load(st_path.to_str().expect("utf8 path"), dev, Kind::Float), dev }
    }
}

/// Sample one of `cand` (policy_index, handle) pairs ~ softmax(logit/temp).
fn sample_cand(cand: &[(usize, u16)], logits: &[f32], temp: f64, rng: &mut StdRng) -> u16 {
    let t = temp.max(1e-6) as f32;
    let maxl = cand.iter().map(|&(i, _)| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = cand.iter().map(|&(i, _)| ((logits[i] - maxl) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let r = rng.gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (k, &(_, h)) in cand.iter().enumerate() {
        acc += exps[k];
        if r <= acc {
            return h;
        }
    }
    cand.last().unwrap().1
}

fn argmax_cand(cand: &[(usize, u16)], logits: &[f32]) -> u16 {
    let mut best = cand[0].1;
    let mut best_v = f32::NEG_INFINITY;
    for &(i, h) in cand {
        if logits[i] > best_v {
            best_v = logits[i];
            best = h;
        }
    }
    best
}

impl Player for NetPlayer {
    fn select_moves(&self, boards: &[&Board], plies: &[usize], opening_plies: usize,
                    rng: &mut StdRng) -> Vec<u16> {
        let m = boards.len();
        let mut enc = vec![0f32; m * ENC_LEN];
        for (j, b) in boards.iter().enumerate() {
            b.encode_into(&mut enc[j * ENC_LEN..(j + 1) * ENC_LEN]);
        }
        let x = Tensor::from_slice(&enc)
            .reshape([m as i64, 18, 8, 8])
            .to_device(self.dev)
            .to_kind(Kind::Float);
        let (logits, _) = tch::no_grad(|| self.net.forward(&x));
        let logits = logits.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
        let mut lv = vec![0f32; m * POLICY_SIZE];
        logits.copy_data(&mut lv, m * POLICY_SIZE);

        boards
            .iter()
            .zip(plies)
            .enumerate()
            .map(|(j, (b, &ply))| {
                let row = &lv[j * POLICY_SIZE..(j + 1) * POLICY_SIZE];
                let cand: Vec<(usize, u16)> = b
                    .legal_moves_vec()
                    .into_iter()
                    .map(|mv| (b.move_to_index(mv) as usize, pack_move(mv)))
                    .collect();
                if ply < opening_plies {
                    sample_cand(&cand, row, OPENING_TEMP, rng)
                } else {
                    argmax_cand(&cand, row)
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Match play (batched across games; alternating colors; W/D/L)
// ---------------------------------------------------------------------------
struct MatchGame {
    board: Board,
    a_is_white: bool,
    done: bool,
    ply: usize,
    rw: i8, // white-POV result
}

fn play_match(player_a: &dyn Player, player_b: &dyn Player, n_games: usize,
              opening_plies: usize, max_plies: usize, rng: &mut StdRng) -> (i32, i32, i32) {
    let mut games: Vec<MatchGame> = (0..n_games)
        .map(|i| MatchGame { board: Board::new(), a_is_white: i % 2 == 0, done: false, ply: 0, rw: 0 })
        .collect();

    while games.iter().any(|g| !g.done) {
        let mut a_idx: Vec<usize> = Vec::new();
        let mut b_idx: Vec<usize> = Vec::new();
        for (gi, g) in games.iter().enumerate() {
            if g.done {
                continue;
            }
            if g.board.turn() == g.a_is_white {
                a_idx.push(gi);
            } else {
                b_idx.push(gi);
            }
        }

        for (player, group) in [(player_a, &a_idx), (player_b, &b_idx)] {
            if group.is_empty() {
                continue;
            }
            let moves = {
                let boards: Vec<&Board> = group.iter().map(|&gi| &games[gi].board).collect();
                let plies: Vec<usize> = group.iter().map(|&gi| games[gi].ply).collect();
                player.select_moves(&boards, &plies, opening_plies, rng)
            };
            for (k, &gi) in group.iter().enumerate() {
                games[gi].board.push(moves[k]);
                games[gi].ply += 1;
                let res = games[gi].board.outcome_white();
                if res.is_some() || games[gi].ply >= max_plies {
                    games[gi].rw = res.unwrap_or(0); // ply cap => draw
                    games[gi].done = true;
                }
            }
        }
    }

    let (mut w, mut d, mut l) = (0, 0, 0);
    for g in &games {
        let a = if g.a_is_white { g.rw } else { -g.rw };
        if a > 0 {
            w += 1;
        } else if a < 0 {
            l += 1;
        } else {
            d += 1;
        }
    }
    (w, d, l)
}

// ---------------------------------------------------------------------------
// Elo fit (Bradley-Terry, coordinate-Newton; random pinned, L2-regularized).
// Copied from eval.rs — score_a counts draws as 0.5.
// ---------------------------------------------------------------------------
fn fit_elo(names: &[String], games: &[(String, String, f64, i64)],
           fixed: &HashMap<String, f64>, reg: f64, iters: usize) -> HashMap<String, f64> {
    let mut r: HashMap<String, f64> =
        names.iter().map(|n| (n.clone(), *fixed.get(n).unwrap_or(&0.0))).collect();
    let q = 10f64.ln() / 400.0;

    let mut adj: HashMap<String, Vec<(String, f64, f64)>> =
        names.iter().map(|n| (n.clone(), Vec::new())).collect();
    for (a, b, score_a, n) in games {
        if *n <= 0 {
            continue;
        }
        let nf = *n as f64;
        adj.get_mut(a).unwrap().push((b.clone(), *score_a, nf));
        adj.get_mut(b).unwrap().push((a.clone(), nf - *score_a, nf));
    }

    for _ in 0..iters {
        for p in names {
            if fixed.contains_key(p) {
                continue;
            }
            let mut g = 0.0;
            let mut h = 0.0;
            let rp = r[p];
            for (opp, score_p, n) in &adj[p] {
                let e = 1.0 / (1.0 + 10f64.powf((r[opp] - rp) / 400.0));
                g += q * (score_p - n * e);
                h += q * q * n * e * (1.0 - e);
            }
            g -= reg * rp;
            h += reg;
            if h > 1e-12 {
                *r.get_mut(p).unwrap() += g / h;
            }
        }
    }
    for (n, v) in fixed {
        r.insert(n.clone(), *v);
    }
    r
}

// ---------------------------------------------------------------------------
// Pool (on-disk JSON). No serving fields — promotion stays manual.
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Clone, Default)]
struct ModelEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    st: Option<String>, // safetensors path relative to run-dir (None for random)
    #[serde(default)]
    rating: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
struct MatchResult {
    a: String,
    b: String,
    score_a: f64,
    n: i64,
}

#[derive(Serialize, Deserialize, Default)]
struct Pool {
    #[serde(default)]
    models: IndexMap<String, ModelEntry>,
    #[serde(default)]
    results: Vec<MatchResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    top: Vec<String>,
}

fn load_pool(path: &Path) -> Pool {
    if path.exists() {
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
    } else {
        Pool::default()
    }
}

fn save_pool(path: &Path, pool: &Pool) {
    let tmp = path.with_extension("tmp");
    let text = serde_json::to_string_pretty(pool).expect("serialize pool");
    fs::write(&tmp, text).unwrap_or_else(|e| panic!("write {tmp:?}: {e}"));
    fs::rename(&tmp, path).unwrap_or_else(|e| panic!("rename {tmp:?} -> {path:?}: {e}"));
}

/// Active opponents = top `n_top` rated + random + `n_anchors` frozen snapshots
/// whose ratings most evenly cover (0, rating of the n_top-th).
fn select_anchors(ratings: &HashMap<String, f64>, models: &[String], n_top: usize,
                  n_anchors: usize) -> (Vec<String>, Vec<String>) {
    let rget = |m: &str| ratings.get(m).copied().unwrap_or(0.0);

    let mut ranked: Vec<String> = models.iter().filter(|m| *m != RANDOM).cloned().collect();
    ranked.sort_by(|a, b| rget(b).partial_cmp(&rget(a)).unwrap());

    let top: Vec<String> = ranked.iter().take(n_top).cloned().collect();
    let mut active = top.clone();
    active.push(RANDOM.to_string());

    let below: Vec<String> = ranked.iter().skip(n_top).cloned().collect();
    if !below.is_empty() {
        let ceiling = if let Some(last_top) = top.last() {
            rget(last_top)
        } else {
            below.iter().map(|m| rget(m)).fold(f64::NEG_INFINITY, f64::max)
        };
        let k = n_anchors.min(below.len());
        let mut pool: Vec<String> = below;
        for i in 0..k {
            let t = ceiling * (i + 1) as f64 / (k + 1) as f64;
            let bi = pool
                .iter()
                .enumerate()
                .min_by(|(_, m1), (_, m2)| {
                    (rget(m1) - t).abs().partial_cmp(&(rget(m2) - t).abs()).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();
            active.push(pool.remove(bi));
        }
    }
    (active, top)
}

fn snapshot_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run_dir = PathBuf::from(flag(&args, "--run-dir", "runs/run1"));
    let candidate = PathBuf::from(flag(&args, "--candidate", ""));
    let games: usize = flag(&args, "--games", "60").parse().unwrap();
    let n_top: usize = flag(&args, "--n-top", "2").parse().unwrap();
    let n_anchors: usize = flag(&args, "--n-anchors", "3").parse().unwrap();
    let opening_plies: usize = flag(&args, "--opening-plies", "8").parse().unwrap();
    let max_plies: usize = flag(&args, "--max-plies", "200").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();
    assert!(!candidate.as_os_str().is_empty(), "--candidate required");

    let dev = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    let pool_path = run_dir.join("pool.json");
    let mut pool = load_pool(&pool_path);
    let mut rng = StdRng::seed_from_u64(seed);

    let cand_name = snapshot_stem(&candidate);
    println!("[eval] candidate={cand_name} device={dev:?}");

    let rel = |p: &Path| -> String {
        p.strip_prefix(&run_dir)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| p.to_string_lossy().into_owned())
    };
    pool.models.insert(cand_name.clone(), ModelEntry { st: Some(rel(&candidate)), rating: None });
    pool.models
        .entry(RANDOM.to_string())
        .or_insert(ModelEntry { st: None, rating: Some(0.0) });

    let prior: HashMap<String, f64> =
        pool.models.iter().map(|(m, e)| (m.clone(), e.rating.unwrap_or(0.0))).collect();
    let model_names: Vec<String> = pool.models.keys().cloned().collect();
    let (active, _) = select_anchors(&prior, &model_names, n_top, n_anchors);
    let mut opponents: Vec<String> = active.into_iter().filter(|m| *m != cand_name).collect();
    if opponents.is_empty() {
        opponents = vec![RANDOM.to_string()];
    }
    println!("[eval] opponents={opponents:?}");

    let cand = NetPlayer::load(&candidate, dev);

    let opp_specs: Vec<(String, Option<String>)> = opponents
        .iter()
        .map(|o| {
            let st = if o == RANDOM {
                None
            } else {
                Some(pool.models[o].st.clone().unwrap_or_else(|| panic!("model {o} has no st path")))
            };
            (o.clone(), st)
        })
        .collect();

    let mut cand_vs_random: Option<f64> = None;
    for (opp, st) in &opp_specs {
        let opp_player: Box<dyn Player> = match st {
            None => Box::new(RandomPlayer),
            Some(p) => Box::new(NetPlayer::load(&run_dir.join(p), dev)),
        };
        let (w, d, l) = play_match(&cand, &*opp_player, games, opening_plies, max_plies, &mut rng);
        let score_a = w as f64 + 0.5 * d as f64;
        let n = (w + d + l) as i64;
        pool.results.push(MatchResult { a: cand_name.clone(), b: opp.clone(), score_a, n });
        if opp == RANDOM {
            cand_vs_random = Some(score_a / n.max(1) as f64);
        }
        println!("[eval] {cand_name} vs {opp}: W{w} D{d} L{l}");
    }

    // Refit Elo globally over all accumulated results (random pinned at 0).
    let names: Vec<String> = pool.models.keys().cloned().collect();
    let game_rows: Vec<(String, String, f64, i64)> =
        pool.results.iter().map(|r| (r.a.clone(), r.b.clone(), r.score_a, r.n)).collect();
    let fixed: HashMap<String, f64> = HashMap::from([(RANDOM.to_string(), 0.0)]);
    let ratings = fit_elo(&names, &game_rows, &fixed, 1e-4, 400);
    for m in &names {
        pool.models.get_mut(m).unwrap().rating = Some(round1(ratings[m]));
    }

    let (_, top) = select_anchors(&ratings, &names, n_top, n_anchors);
    pool.top = top;

    let mut ranked = names.clone();
    ranked.sort_by(|a, b| ratings[b].partial_cmp(&ratings[a]).unwrap());
    let rating_str = ranked
        .iter()
        .map(|m| format!("{m}={}", pool.models[m].rating.unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join(", ");
    println!("[eval] ratings: {rating_str}");

    let wr = cand_vs_random.map(|c| 100.0 * c).unwrap_or(f64::NAN);
    println!("[eval] {cand_name} elo={:.0} score_vs_random={:.1}%", ratings[&cand_name], wr);

    save_pool(&pool_path, &pool);
    println!("[eval] done");
}
