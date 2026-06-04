//! Rust port of `training/evaluate.py` — self-play evaluation, Elo rating, and
//! model serving. Domain-general, purely self-play: a candidate snapshot plays
//! match games against an active pool (top performers + a fixed random anchor +
//! frozen ex-snapshot anchors), a lightweight Bradley-Terry Elo is refit over
//! all accumulated results (random pinned at 0), the pool is updated, and the
//! highest-rated model is promoted to `<run-dir>/best.onnx` if it clears the
//! served model by a confidence margin.
//!
//! Search reuses `chess_core` (board + MCTS primitives); the net forward is the
//! eager parity-exact fp32 path (`net::Net::forward_unfolded`). Eval is low
//! throughput, so this is a straightforward synchronous loop — none of the
//! self-play worker's AOTI / CUDA-graph / async-pipeline machinery.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use chess_core::board::{pack_move, Board};
use chess_core::mcts::{
    backprop, expand_node, sample_move, walk_to_leaf, Node, ENC_LEN, MAX_PLIES, POLICY_SIZE,
};
use indexmap::IndexMap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use crate::net::Net;

const RANDOM: &str = "random"; // reserved name for the fixed floor anchor (Elo pinned at 0)
const OPENING_TEMP: f64 = 1.0; // temperature for the opening plies, for game diversity

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

// ---------------------------------------------------------------------------
// Net evaluation (batched fp32 forward over a stack of encodings)
// ---------------------------------------------------------------------------
/// Run `net` over `m` encodings (`enc` is `m * ENC_LEN` f32) and return
/// `(logits[m*POLICY_SIZE], values[m])` on the host.
fn eval_batch(net: &Net, dev: Device, enc: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
    let x = Tensor::from_slice(enc)
        .reshape([m as i64, 18, 8, 8])
        .to_device(dev)
        .to_kind(Kind::Float);
    let (logits, values) = tch::no_grad(|| net.forward_unfolded(&x));
    let lc = logits.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let vc = values.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let mut lv = vec![0f32; m * POLICY_SIZE];
    lc.copy_data(&mut lv, m * POLICY_SIZE);
    let mut vv = vec![0f32; m];
    vc.copy_data(&mut vv, m);
    (lv, vv)
}

// ---------------------------------------------------------------------------
// Synchronous batched MCTS (port of mcts.py:run_simulations, noise off)
// ---------------------------------------------------------------------------
struct Holder<'b> {
    board: &'b Board, // the real game board (never mutated; clones are searched)
    arena: Vec<Node>, // root at index 0
}

struct Leaf {
    holder: usize,
    path: Vec<usize>,
    board: Board, // the repetition-blind search clone walked to the leaf
    tval: Option<f32>,
}

/// Advance every holder's tree by `sims` simulations, batching leaf evaluations
/// across all holders through one net forward per step. No root noise (eval).
fn run_simulations(holders: &mut [Holder], net: &Net, dev: Device, sims: usize) {
    // Ensure each root is expanded (single batched pass).
    let need: Vec<usize> = (0..holders.len())
        .filter(|&i| !holders[i].arena[0].expanded)
        .collect();
    if !need.is_empty() {
        let mut enc = vec![0f32; need.len() * ENC_LEN];
        for (j, &i) in need.iter().enumerate() {
            holders[i].board.encode_into(&mut enc[j * ENC_LEN..(j + 1) * ENC_LEN]);
        }
        let (logits, _) = eval_batch(net, dev, &enc, need.len());
        for (j, &i) in need.iter().enumerate() {
            let board = holders[i].board.clone_search();
            expand_node(
                &mut holders[i].arena,
                0,
                &board,
                &logits[j * POLICY_SIZE..(j + 1) * POLICY_SIZE],
            );
        }
    }

    for _ in 0..sims {
        // Gather one leaf per holder (walk a repetition-blind clone).
        let mut leaves: Vec<Leaf> = Vec::with_capacity(holders.len());
        for hi in 0..holders.len() {
            let mut board = holders[hi].board.clone_search();
            let path = walk_to_leaf(&holders[hi].arena, 0, &mut board);
            let tval = board.terminal_value();
            leaves.push(Leaf { holder: hi, path, board, tval });
        }

        // Batch-evaluate the non-terminal leaves.
        let eval_idxs: Vec<usize> = (0..leaves.len()).filter(|&k| leaves[k].tval.is_none()).collect();
        let (logits, values) = if eval_idxs.is_empty() {
            (None, None)
        } else {
            let mut enc = vec![0f32; eval_idxs.len() * ENC_LEN];
            for (j, &k) in eval_idxs.iter().enumerate() {
                leaves[k].board.encode_into(&mut enc[j * ENC_LEN..(j + 1) * ENC_LEN]);
            }
            let (l, v) = eval_batch(net, dev, &enc, eval_idxs.len());
            (Some(l), Some(v))
        };

        // Scatter: expand non-terminal leaves, backprop terminal/value scores.
        let mut ei = 0usize;
        for leaf in &leaves {
            let v = match leaf.tval {
                Some(t) => t as f64,
                None => {
                    let logits = logits.as_ref().unwrap();
                    let leaf_idx = *leaf.path.last().unwrap();
                    expand_node(
                        &mut holders[leaf.holder].arena,
                        leaf_idx,
                        &leaf.board,
                        &logits[ei * POLICY_SIZE..(ei + 1) * POLICY_SIZE],
                    );
                    let value = values.as_ref().unwrap()[ei] as f64;
                    ei += 1;
                    value
                }
            };
            backprop(&mut holders[leaf.holder].arena, &leaf.path, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Players
// ---------------------------------------------------------------------------
trait Player {
    /// Pick a move for each board (the group of games where this player is to
    /// move). `plies[i]` is board `i`'s current ply (for the opening-temp gate).
    fn select_moves(
        &self,
        boards: &[&Board],
        plies: &[usize],
        opening_plies: usize,
        rng: &mut StdRng,
    ) -> Vec<u16>;
}

struct RandomPlayer;

impl Player for RandomPlayer {
    fn select_moves(
        &self,
        boards: &[&Board],
        _plies: &[usize],
        _opening_plies: usize,
        rng: &mut StdRng,
    ) -> Vec<u16> {
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
    sims: usize,
}

impl NetPlayer {
    fn load(st_path: &Path, sims: usize, dev: Device) -> NetPlayer {
        let net = Net::load(st_path.to_str().expect("non-utf8 path"), dev, Kind::Float);
        NetPlayer { net, dev, sims }
    }
}

impl Player for NetPlayer {
    fn select_moves(
        &self,
        boards: &[&Board],
        plies: &[usize],
        opening_plies: usize,
        rng: &mut StdRng,
    ) -> Vec<u16> {
        // Fresh tree per move (no reuse) — simpler and fine for eval volumes.
        let mut holders: Vec<Holder> = boards
            .iter()
            .map(|&b| Holder { board: b, arena: vec![Node::new(0.0, -1)] })
            .collect();
        run_simulations(&mut holders, &self.net, self.dev, self.sims);
        holders
            .iter()
            .zip(plies)
            .map(|(h, &ply)| {
                let temp = if ply < opening_plies { OPENING_TEMP } else { 0.0 };
                sample_move(&h.arena, 0, temp, rng)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Match play
// ---------------------------------------------------------------------------
struct MatchGame {
    board: Board,
    a_is_white: bool,
    done: bool,
    ply: usize,
    rw: i8, // white-POV result
}

/// Play `n_games` between two players, alternating colors. The first
/// `opening_plies` are sampled with temperature for diversity. Net evaluations
/// are batched across all games where the same player is to move. Returns
/// (wins, draws, losses) from `player_a`'s perspective.
fn play_match(
    player_a: &dyn Player,
    player_b: &dyn Player,
    n_games: usize,
    opening_plies: usize,
    rng: &mut StdRng,
) -> (i32, i32, i32) {
    let mut games: Vec<MatchGame> = (0..n_games)
        .map(|i| MatchGame {
            board: Board::new(),
            a_is_white: i % 2 == 0,
            done: false,
            ply: 0,
            rw: 0,
        })
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
            // Scope the immutable borrow of `games` so we can mutate it below.
            let moves = {
                let boards: Vec<&Board> = group.iter().map(|&gi| &games[gi].board).collect();
                let plies: Vec<usize> = group.iter().map(|&gi| games[gi].ply).collect();
                player.select_moves(&boards, &plies, opening_plies, rng)
            };
            for (k, &gi) in group.iter().enumerate() {
                games[gi].board.push(moves[k]);
                games[gi].ply += 1;
                let res = games[gi].board.outcome_white();
                if res.is_some() || games[gi].ply >= MAX_PLIES {
                    games[gi].rw = res.unwrap_or(0);
                    games[gi].done = true;
                }
            }
        }
    }

    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    for g in &games {
        let a = if g.a_is_white { g.rw } else { -g.rw };
        if a > 0 {
            wins += 1;
        } else if a < 0 {
            losses += 1;
        } else {
            draws += 1;
        }
    }
    (wins, draws, losses)
}

// ---------------------------------------------------------------------------
// Elo fit (Bradley-Terry, coordinate-Newton; random pinned, L2-regularized)
// ---------------------------------------------------------------------------
/// `games`: (a, b, score_a, n) — in n games between a and b, a scored score_a
/// points (win=1, draw=0.5). Names in `fixed` are held at their fixed rating to
/// pin the scale; `reg` pulls ratings toward 0 so a 100% sweep stays finite.
fn fit_elo(
    names: &[String],
    games: &[(String, String, f64, i64)],
    fixed: &HashMap<String, f64>,
    reg: f64,
    iters: usize,
) -> HashMap<String, f64> {
    let mut r: HashMap<String, f64> =
        names.iter().map(|n| (n.clone(), *fixed.get(n).unwrap_or(&0.0))).collect();
    let q = 10f64.ln() / 400.0;

    // adjacency: name -> Vec<(opponent, score_for_name, n)>
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
// Pool (on-disk JSON, schema-compatible with evaluate.py except pt -> st)
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Clone, Default)]
struct ModelEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    st: Option<String>, // safetensors path relative to run-dir (was "pt")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    onnx: Option<String>,
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
    served: Option<String>,
    #[serde(default)]
    served_rating: Option<f64>,
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
/// whose ratings most evenly cover (0, rating of the n_top-th). Returns
/// (active opponent set, top list). `models` must be in pool insertion order.
fn select_anchors(
    ratings: &HashMap<String, f64>,
    models: &[String],
    n_top: usize,
    n_anchors: usize,
) -> (Vec<String>, Vec<String>) {
    let rget = |m: &str| ratings.get(m).copied().unwrap_or(0.0);

    let mut ranked: Vec<String> = models.iter().filter(|m| *m != RANDOM).cloned().collect();
    // Stable descending sort by rating (ties keep insertion order, like Python).
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
            // Closest unused model (first on ties, matching Python min()).
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

/// Copy the leader's onnx to best.onnx. Unlike Python, there is no ONNX-export
/// fallback (every snapshot ships an .onnx; random is never a leader) and no
/// best.pt (dropped — the browser serves best.onnx only).
fn serve(run_dir: &Path, pool: &Pool, name: &str) {
    let onnx_rel = pool.models[name]
        .onnx
        .as_ref()
        .unwrap_or_else(|| panic!("model {name} has no onnx to serve"));
    let onnx = run_dir.join(onnx_rel);
    fs::copy(&onnx, run_dir.join("best.onnx"))
        .unwrap_or_else(|e| panic!("copy {onnx:?} -> best.onnx: {e}"));
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    run_dir: &Path,
    candidate: &Path,
    games_per_pair: usize,
    sims: usize,
    opening_plies: usize,
    n_top: usize,
    n_anchors: usize,
    serve_margin: f64,
    seed: u64,
) {
    let pool_path = run_dir.join("pool.json");
    let mut pool = load_pool(&pool_path);
    let mut rng = StdRng::seed_from_u64(seed);
    let dev = Device::cuda_if_available();

    let cand_name = candidate
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("candidate stem")
        .to_string();
    let cand_onnx = candidate.with_extension("onnx");
    println!("[eval] candidate={cand_name} device={dev:?}");

    // Register candidate (paths stored relative to run-dir for portability).
    let rel = |p: &Path| -> String {
        p.strip_prefix(run_dir).map(|r| r.to_string_lossy().into_owned()).unwrap_or_else(|_| p.to_string_lossy().into_owned())
    };
    pool.models.insert(
        cand_name.clone(),
        ModelEntry { st: Some(rel(candidate)), onnx: Some(rel(&cand_onnx)), rating: None },
    );
    pool.models
        .entry(RANDOM.to_string())
        .or_insert(ModelEntry { st: None, onnx: None, rating: Some(0.0) });

    // Pick the active opponent set from prior ratings (top-N + random + anchors).
    let prior: HashMap<String, f64> =
        pool.models.iter().map(|(m, e)| (m.clone(), e.rating.unwrap_or(0.0))).collect();
    let model_names: Vec<String> = pool.models.keys().cloned().collect();
    let (active, _) = select_anchors(&prior, &model_names, n_top, n_anchors);
    let mut opponents: Vec<String> = active.into_iter().filter(|m| *m != cand_name).collect();
    if opponents.is_empty() {
        opponents = vec![RANDOM.to_string()];
    }
    println!("[eval] opponents={opponents:?}");

    let cand_player = NetPlayer::load(candidate, sims, dev);

    // Resolve each opponent's safetensors path out of the pool up front, so the
    // match loop can mutate pool.results without holding a borrow on the pool.
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
            Some(p) => Box::new(NetPlayer::load(&run_dir.join(p), sims, dev)),
        };
        let (w, d, l) = play_match(&cand_player, &*opp_player, games_per_pair, opening_plies, &mut rng);
        let score_a = w as f64 + 0.5 * d as f64;
        pool.results.push(MatchResult {
            a: cand_name.clone(),
            b: opp.clone(),
            score_a,
            n: (w + d + l) as i64,
        });
        if opp == RANDOM {
            cand_vs_random = Some(score_a / (w + d + l).max(1) as f64);
        }
        println!("[eval] {cand_name} vs {opp}: W{w} D{d} L{l}");
    }

    // Refit Elo over all accumulated results.
    let names: Vec<String> = pool.models.keys().cloned().collect();
    let games: Vec<(String, String, f64, i64)> =
        pool.results.iter().map(|r| (r.a.clone(), r.b.clone(), r.score_a, r.n)).collect();
    let fixed: HashMap<String, f64> = HashMap::from([(RANDOM.to_string(), 0.0)]);
    let ratings = fit_elo(&names, &games, &fixed, 1e-4, 400);
    for m in &names {
        pool.models.get_mut(m).unwrap().rating = Some(round1(ratings[m]));
    }

    // Recompute the active set for the next round and decide serving.
    let (_, top) = select_anchors(&ratings, &names, n_top, n_anchors);
    let leader = top.first().cloned();
    pool.top = top;

    let mut ranked_names = names.clone();
    ranked_names.sort_by(|a, b| {
        ratings.get(b).unwrap_or(&0.0).partial_cmp(ratings.get(a).unwrap_or(&0.0)).unwrap()
    });
    let rating_str = ranked_names
        .iter()
        .map(|m| format!("{m}={}", pool.models[m].rating.unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join(", ");
    println!("[eval] ratings: {rating_str}");

    let served = pool.served.clone();
    // Compare against the served model's CURRENT refit rating, not the frozen
    // served_rating: every refit drifts the whole scale, so a stale threshold
    // becomes unbeatable and best.onnx would stick forever.
    let served_now = served.as_ref().and_then(|s| ratings.get(s).copied());
    let sane = cand_vs_random.map_or(true, |c| c >= 0.6);
    if let Some(leader) = leader.filter(|_| sane) {
        let lead_r = ratings[&leader];
        if served.is_none() || lead_r > served_now.unwrap_or(-1e9) + serve_margin {
            serve(run_dir, &pool, &leader);
            pool.served = Some(leader.clone());
            pool.served_rating = Some(round1(lead_r));
            println!("[eval] SERVED {leader} (rating={lead_r:.1})");
        } else {
            println!(
                "[eval] kept {} (rating={:.1}); leader {leader}={lead_r:.1} within margin",
                served.as_deref().unwrap_or("?"),
                served_now.unwrap_or(0.0)
            );
        }
    } else if !sane {
        println!(
            "[eval] candidate failed random sanity gate (score vs random={:.2}); not serving",
            cand_vs_random.unwrap_or(0.0)
        );
    }

    save_pool(&pool_path, &pool);
    println!("[eval] done");
}

/// Test/diagnostic hook: load pool.json, refit Elo over the recorded results,
/// and print `name\trating` lines sorted by rating (no match play). Used by the
/// fit_elo parity check against the Python reference.
pub fn fit_only(run_dir: &Path) {
    let pool = load_pool(&run_dir.join("pool.json"));
    let names: Vec<String> = pool.models.keys().cloned().collect();
    let games: Vec<(String, String, f64, i64)> =
        pool.results.iter().map(|r| (r.a.clone(), r.b.clone(), r.score_a, r.n)).collect();
    let fixed: HashMap<String, f64> = HashMap::from([(RANDOM.to_string(), 0.0)]);
    let ratings = fit_elo(&names, &games, &fixed, 1e-4, 400);
    let mut ns = names.clone();
    ns.sort_by(|a, b| ratings[b].partial_cmp(&ratings[a]).unwrap());
    for n in &ns {
        println!("{n}\t{:.4}", ratings[n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // fit_elo: a 100% sweep stays finite (reg), and order matches strength.
    #[test]
    fn elo_orders_by_strength() {
        let names: Vec<String> = ["random", "weak", "strong"].iter().map(|s| s.to_string()).collect();
        let games = vec![
            ("weak".to_string(), "random".to_string(), 80.0, 100),
            ("strong".to_string(), "random".to_string(), 95.0, 100),
            ("strong".to_string(), "weak".to_string(), 70.0, 100),
        ];
        let fixed = HashMap::from([("random".to_string(), 0.0)]);
        let r = fit_elo(&names, &games, &fixed, 1e-4, 400);
        assert!((r["random"]).abs() < 1e-9);
        assert!(r["strong"] > r["weak"], "strong {} should beat weak {}", r["strong"], r["weak"]);
        assert!(r["weak"] > 0.0, "weak {} should beat random", r["weak"]);
    }
}
