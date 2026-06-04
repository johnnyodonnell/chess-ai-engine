//! evaluate_rs — the 100% Rust port of `training/evaluate.py`.
//!
//! Plays a candidate snapshot against the active pool, refits Elo, updates
//! pool.json, and promotes the leader to best.onnx. The orchestrator launches
//! this after each snapshot (replacing the old `python evaluate.py`).
//!
//!   evaluate_rs --run-dir <dir> --candidate <snap>.safetensors [flags]

use std::path::PathBuf;
use std::process::exit;

/// Read a `--key value` flag, falling back to `default`.
fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn require(args: &[String], key: &str) -> String {
    let v = flag(args, key, "");
    if v.is_empty() {
        eprintln!("missing required {key}");
        eprintln!("usage: evaluate_rs --run-dir <dir> --candidate <snap>.safetensors \\");
        eprintln!("  [--games-per-pair 100] [--sims 200] [--opening-plies 10] \\");
        eprintln!("  [--n-top 2] [--n-anchors 3] [--serve-margin 35] [--seed 0]");
        exit(2);
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run_dir = PathBuf::from(require(&args, "--run-dir"));

    // Diagnostic: refit Elo over the recorded pool results and print ratings
    // (no match play). Used by the fit_elo parity check.
    if args.iter().any(|a| a == "--fit-only") {
        selfplay_rs::eval::fit_only(&run_dir);
        return;
    }

    let candidate = PathBuf::from(require(&args, "--candidate"));

    selfplay_rs::eval::evaluate(
        &run_dir,
        &candidate,
        flag(&args, "--games-per-pair", "100").parse().unwrap(),
        flag(&args, "--sims", "200").parse().unwrap(),
        flag(&args, "--opening-plies", "10").parse().unwrap(),
        flag(&args, "--n-top", "2").parse().unwrap(),
        flag(&args, "--n-anchors", "3").parse().unwrap(),
        flag(&args, "--serve-margin", "35").parse().unwrap(),
        flag(&args, "--seed", "0").parse().unwrap(),
    );
}
