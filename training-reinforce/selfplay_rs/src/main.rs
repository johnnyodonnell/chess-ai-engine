//! selfplay_reinforce CLI.
//!
//! Subcommands:
//!   forward-check <dir>   Parity gate: Rust tch forward vs a PyTorch fixture
//!                         (fwd_weights.safetensors + fwd_fixture.safetensors in <dir>)
//!   selfplay …            Generate one cohort to a file (debug / smoke test)
//!   selfplay-serve        Persistent worker: CUDA init once, one cohort per
//!                         stdin JSON command (driven by orchestrator.py)

use std::collections::HashMap;
use std::io::{BufRead, Write};

use serde::Deserialize;
use tch::{Device, Kind, Tensor};

use selfplay_reinforce::net::Net;
use selfplay_reinforce::pipeline;
use selfplay_reinforce::selfplay::{self, Config};

fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn read_fixture(path: &str) -> HashMap<String, Tensor> {
    Tensor::read_safetensors(path)
        .unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
        .into_iter()
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.to_kind(Kind::Float).to_device(Device::Cpu);
    let b = b.to_kind(Kind::Float).to_device(Device::Cpu);
    (a - b).abs().max().double_value(&[])
}

fn forward_check(dir: &str) -> bool {
    let wpath = format!("{dir}/fwd_weights.safetensors");
    let fix = read_fixture(&format!("{dir}/fwd_fixture.safetensors"));
    let input = fix.get("input").expect("fixture.input"); // [n,18,8,8]
    let ref_logits = fix.get("ref_logits").expect("fixture.ref_logits"); // [n,4672]
    let ref_value = fix.get("ref_value").expect("fixture.ref_value"); // [n]
    let n = input.size()[0];

    // ---- CPU fp32: exact-math gate vs PyTorch CPU fp32 (no TF32 noise) ----
    let net_cpu = Net::load(&wpath, Device::Cpu, Kind::Float);
    let x_cpu = input.to_device(Device::Cpu).to_kind(Kind::Float);
    let (pl, vl) = net_cpu.forward(&x_cpu);
    let dl = max_abs_diff(&pl, ref_logits);
    let dv = max_abs_diff(&vl, ref_value);
    println!("forward-check on {n} positions:");
    println!("  CPU fp32 vs PyTorch: max|Δlogits|={dl:.3e}  max|Δvalue|={dv:.3e}");
    let cpu_ok = dl < 1e-4 && dv < 1e-4;

    // ---- GPU smoke: prove the CUDA path runs and is close ----
    let mut gpu_ok = true;
    if tch::Cuda::is_available() {
        let dev = Device::Cuda(0);
        let net_g = Net::load(&wpath, dev, Kind::Float);
        let xg = input.to_device(dev).to_kind(Kind::Float);
        let (plg, vlg) = net_g.forward(&xg);
        let dlg = max_abs_diff(&plg, ref_logits);
        let dvg = max_abs_diff(&vlg, ref_value);
        println!("  GPU fp32 vs PyTorch:  max|Δlogits|={dlg:.3e}  max|Δvalue|={dvg:.3e}");
        // fp32 GPU may use TF32 — sanity bounds only.
        gpu_ok = dlg < 5e-2 && dvg < 5e-2;
    } else {
        println!("  (CUDA unavailable — skipping GPU smoke)");
    }

    if !cpu_ok {
        println!("  FAIL: CPU fp32 diverges from PyTorch (math/layout bug)");
    }
    if !gpu_ok {
        println!("  FAIL: GPU forward outside sanity bounds");
    }
    cpu_ok && gpu_ok
}

fn default_threads() -> usize {
    16
}

#[derive(Deserialize)]
struct Command {
    weights: String,
    out: String,
    target_rows: usize,
    batch: usize,
    #[serde(default = "default_threads")]
    threads: usize,
    #[serde(default)]
    concurrency: usize, // 0 => 2*batch
    temperature: f64,
    temp_end: f64,
    max_plies: u32,
    seed: u64,
    #[serde(default)]
    cpu: bool,
}

/// Persistent worker. CUDA/libtorch init once; one cohort per stdin line. The
/// protocol channel is stdout (one JSON line per message) — all stats/logs go to
/// stderr so they never corrupt it.
fn serve(cpu: bool) {
    let dev = selfplay::pick_device(cpu);
    let stdout = std::io::stdout();
    {
        let mut o = stdout.lock();
        writeln!(o, "{}", serde_json::json!({"ready": true})).unwrap();
        o.flush().unwrap();
    }
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let cmd: Command = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("bad command {line:?}: {e}"));
        let cfg = Config {
            weights: cmd.weights,
            out: cmd.out,
            target_rows: cmd.target_rows,
            batch: cmd.batch,
            threads: cmd.threads,
            concurrency: cmd.concurrency,
            temperature: cmd.temperature,
            temp_end: cmd.temp_end,
            max_plies: cmd.max_plies,
            seed: cmd.seed,
            cpu: cmd.cpu,
        };
        let n = pipeline::run_on(&cfg, dev);
        let mut o = stdout.lock();
        writeln!(o, "{}", serde_json::json!({"done": true, "rows": n})).unwrap();
        o.flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "forward-check" => {
            let dir = args.get(2).map(String::as_str).unwrap_or(".");
            let ok = forward_check(dir);
            println!("{}", if ok { "FORWARD-CHECK OK" } else { "FORWARD-CHECK FAILED" });
            std::process::exit(if ok { 0 } else { 1 });
        }
        "selfplay" => {
            let temperature: f64 = flag(&args, "--temperature", "1.0").parse().unwrap();
            let cfg = Config {
                weights: flag(&args, "--weights", "weights.safetensors"),
                out: flag(&args, "--out", "cohort.bin"),
                target_rows: flag(&args, "--target-rows", "8192").parse().unwrap(),
                batch: flag(&args, "--batch", "512").parse().unwrap(),
                threads: flag(&args, "--threads", "16").parse().unwrap(),
                concurrency: flag(&args, "--concurrency", "0").parse().unwrap(),
                temperature,
                temp_end: flag(&args, "--temp-end", &temperature.to_string()).parse().unwrap(),
                max_plies: flag(&args, "--max-plies", "200").parse().unwrap(),
                seed: flag(&args, "--seed", "0").parse().unwrap(),
                cpu: args.iter().any(|a| a == "--cpu"),
            };
            pipeline::run(&cfg);
        }
        "selfplay-serve" => {
            serve(args.iter().any(|a| a == "--cpu"));
        }
        other => {
            eprintln!("unknown subcommand {other:?}; expected: forward-check | selfplay | selfplay-serve");
            std::process::exit(2);
        }
    }
}
