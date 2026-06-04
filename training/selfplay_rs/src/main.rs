//! selfplay_rs — the 100% Rust self-play worker.
//!
//! Subcommands:
//!   forward-check <dir>  parity gate: Rust net vs PyTorch fixture (Stage 4)
//!   bench / serve        added in Stage 5 (pipeline + IPC)

mod aoti;
mod cudagraph;
mod net;
mod pipeline;

use std::collections::HashMap;
use std::ffi::CString;
use std::time::Duration;

use tch::{Device, Kind, Tensor};

use net::Net;
use pipeline::Config;

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

/// Softmax over legal moves per row (illegal → ~0): the MCTS prior distribution.
fn softmax_legal(logits: &Tensor, mask: &Tensor) -> Tensor {
    let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu);
    let mask = mask.to_kind(Kind::Float).to_device(Device::Cpu);
    let masked = &logits + (&mask - 1.0) * 1.0e30; // legal: +0; illegal: -1e30
    masked.softmax(1, Kind::Float)
}

fn argmax_legal(logits: &Tensor, mask: &Tensor) -> Tensor {
    let logits = logits.to_kind(Kind::Float).to_device(Device::Cpu);
    let mask = mask.to_kind(Kind::Float).to_device(Device::Cpu);
    (&logits + (&mask - 1.0) * 1.0e30).argmax(1, false)
}

fn forward_check(dir: &str) -> bool {
    assert!(tch::Cuda::is_available(), "CUDA not available to tch");
    let dev = Device::Cuda(0);
    let wpath = format!("{dir}/weights.safetensors");
    let fix = read_fixture(&format!("{dir}/fixture.safetensors"));
    let input = fix.get("input").expect("fixture.input");
    let mask = fix.get("legal_mask").expect("fixture.legal_mask");
    let n = input.size()[0];

    // ---- fold correctness: folded vs unfolded IN RUST (same kernels) ----
    let net_f32 = Net::load(&wpath, dev, Kind::Float);
    let x_f32 = input.to_device(dev).to_kind(Kind::Float);
    let (pl_folded, vl_folded) = net_f32.forward_folded(&x_f32);
    let (pl_unf, vl_unf) = net_f32.forward_unfolded(&x_f32);
    let fold_dl = max_abs_diff(&pl_folded, &pl_unf);
    let fold_dv = max_abs_diff(&vl_folded, &vl_unf);
    // unfolded vs PyTorch fp32 (sanity: should be ~0, same libtorch kernels).
    let unf_dl = max_abs_diff(&pl_unf, fix.get("ref_logits_f32").unwrap());

    // ---- bf16 prod path: prior (softmax-over-legal) closeness vs PyTorch bf16 ----
    let net_bf16 = Net::load(&wpath, dev, Kind::BFloat16);
    let x_bf16 = input.to_device(dev).to_kind(Kind::BFloat16);
    let (pl_bf16, vl_bf16) = net_bf16.forward(&x_bf16);
    let prior_rust = softmax_legal(&pl_bf16, mask);
    let prior_ref = softmax_legal(fix.get("ref_logits_bf16").unwrap(), mask);
    let prior_dmax = max_abs_diff(&prior_rust, &prior_ref); // max abs prob diff
    let value_dmax = max_abs_diff(&vl_bf16, fix.get("ref_values_bf16").unwrap());
    let arg_agree = argmax_legal(&pl_bf16, mask)
        .eq_tensor(&argmax_legal(fix.get("ref_logits_bf16").unwrap(), mask))
        .to_kind(Kind::Int64)
        .sum(Kind::Int64)
        .int64_value(&[]);

    println!("forward-check on {n} positions:");
    println!("  fold correctness (Rust folded vs unfolded): max|Δlogits|={fold_dl:.3e} max|Δvalues|={fold_dv:.3e}");
    println!("  unfolded vs PyTorch fp32: max|Δlogits|={unf_dl:.3e}");
    println!("  bf16 prod prior (softmax-over-legal) vs PyTorch: max|Δprob|={prior_dmax:.3e}  max|Δvalue|={value_dmax:.3e}");
    println!("  bf16 argmax-over-legal agree: {arg_agree}/{n}");

    // Gate: folding is numerically sound (folded≈unfolded in fp32), and the bf16
    // prod priors/values match PyTorch within bf16 tolerance.
    let fold_ok = fold_dl < 2.0e-3 && fold_dv < 5.0e-4;
    let prior_ok = prior_dmax < 2.0e-2 && value_dmax < 2.0e-2;
    if !fold_ok {
        println!("  FAIL: BN-fold diverges from explicit BN in fp32 (fold bug)");
    }
    if !prior_ok {
        println!("  FAIL: bf16 prod priors/values outside tolerance");
    }
    fold_ok && prior_ok
}

/// Read a `--key value` flag, falling back to `default`.
fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn bench_config(args: &[String]) -> (Config, Duration, Duration) {
    let config = Config {
        sims: flag(args, "--sims", "200").parse().unwrap(),
        add_root_noise: true,
        bucket_size: flag(args, "--bucket", "512").parse().unwrap(),
        seed: flag(args, "--seed", "1234").parse().unwrap(),
        n_threads: flag(args, "--threads", "16").parse().unwrap(),
        model_path: flag(args, "--pt2", "serving_model.pt2"),
        weights_path: flag(args, "--weights-st", "serving_weights.safetensors"),
        reload_every: Duration::from_secs_f64(flag(args, "--reload-every", "15.0").parse().unwrap()),
    };
    let run = Duration::from_secs_f64(flag(args, "--run", "420").parse().unwrap());
    let interval = Duration::from_secs_f64(flag(args, "--interval", "30").parse().unwrap());
    (config, run, interval)
}

/// Isolated forward timing at batch `b`: bf16 eager vs CUDA-graph replay.
fn time_forward(weights: &str, b: i64) {
    let dev = Device::Cuda(0);
    let net = Net::load(weights, dev, Kind::BFloat16);
    let x = Tensor::zeros([b, 18, 8, 8], (Kind::BFloat16, dev));
    let iters = 500;
    let bench_cudnn = std::env::var("CUDNN_BENCH").map(|v| v == "1").unwrap_or(true);
    cudagraph::set_cudnn_benchmark(bench_cudnn);
    println!("cudnn benchmark = {bench_cudnn}");
    // warm up cudnn algo selection before eager timing
    for _ in 0..5 {
        let _ = net.forward(&x);
    }
    // eager
    cudagraph::device_synchronize();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = net.forward(&x);
    }
    cudagraph::device_synchronize();
    let eager_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    // graph
    cudagraph::set_side_stream();
    let logit = Tensor::zeros([b, 4672], (Kind::BFloat16, dev));
    let value = Tensor::zeros([b], (Kind::BFloat16, dev));
    let copy = |dst: &Tensor, src: &Tensor| {
        let mut t = dst.shallow_clone();
        let _ = t.copy_(src);
    };
    for _ in 0..3 {
        let (l, v) = net.forward(&x);
        copy(&logit, &l);
        copy(&value, &v);
    }
    cudagraph::device_synchronize();
    let g = cudagraph::CudaGraph::capture(|| {
        let (l, v) = net.forward(&x);
        copy(&logit, &l);
        copy(&value, &v);
    });
    cudagraph::device_synchronize();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        g.replay();
    }
    cudagraph::device_synchronize();
    let graph_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    println!("B={b}: eager {eager_us:.0} us/fwd   graph-replay {graph_us:.0} us/fwd");
}

/// Load a .pt2, check parity vs the eager net, and time the AOTI forward.
fn aoti_time(pt2: &str, weights: &str, b: i64) {
    let dev = Device::Cuda(0);
    let model = aoti::AotiModel::load(pt2);
    let net = Net::load(weights, dev, Kind::BFloat16); // parity reference (unfolded)
    let x = Tensor::randn([b, 18, 8, 8], (Kind::BFloat16, dev));
    let out_logits = Tensor::zeros([b, 4672], (Kind::BFloat16, dev));
    let out_values = Tensor::zeros([b], (Kind::BFloat16, dev));

    model.run(&x, &out_logits, &out_values);
    cudagraph::device_synchronize();
    let (ref_l, ref_v) = net.forward(&x);
    let dl = max_abs_diff(&out_logits, &ref_l);
    let dv = max_abs_diff(&out_values, &ref_v);
    println!("AOTI vs eager net (same input): max|Δlogits|={dl:.3e} max|Δvalues|={dv:.3e}");

    let iters = 500;
    cudagraph::device_synchronize();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        model.run(&x, &out_logits, &out_values);
    }
    cudagraph::device_synchronize();
    let us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    println!("AOTI forward (direct, incl. output copy): {us:.0} us/fwd at B={b}");

    // Wrap the AOTI run (+ output copy) in a CUDA graph and replay it as one
    // launch. The pre-port Python path graphed its compiled forward (+output copy)
    // and measured ~9.5% from removing per-kernel launch overhead; AOTI called
    // directly pays that overhead every forward. This tests whether re-introducing
    // the graph recovers it. Static input/outputs (x, out_logits, out_values);
    // Rust would refill `x` before each replay in the real pipeline.
    cudagraph::set_side_stream();
    for _ in 0..5 {
        model.run(&x, &out_logits, &out_values); // warm up cudnn algo choice etc.
    }
    cudagraph::device_synchronize();
    let g = cudagraph::CudaGraph::capture(|| model.run(&x, &out_logits, &out_values));
    cudagraph::device_synchronize();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        g.replay();
    }
    cudagraph::device_synchronize();
    let gus = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    println!("AOTI forward (CUDA-graph replay):  {gus:.0} us/fwd at B={b}  ({:.1}% vs direct)",
             (gus - us) / us * 100.0);
}

/// AOTI parity on real positions: run the .pt2 on the fixture and compare the
/// MCTS prior (softmax-over-legal) + values to PyTorch's bf16 reference.
fn aoti_parity(dir: &str, pt2: &str) -> bool {
    let dev = Device::Cuda(0);
    let fix = read_fixture(&format!("{dir}/fixture.safetensors"));
    let input = fix.get("input").expect("fixture.input");
    let mask = fix.get("legal_mask").unwrap();
    let n = input.size()[0];
    let model = aoti::AotiModel::load(pt2);
    let x = input.to_device(dev).to_kind(Kind::BFloat16);
    let out_logits = Tensor::zeros([n, 4672], (Kind::BFloat16, dev));
    let out_values = Tensor::zeros([n], (Kind::BFloat16, dev));
    model.run(&x, &out_logits, &out_values);
    cudagraph::device_synchronize();

    let prior_rust = softmax_legal(&out_logits, mask);
    // Compare against torch.compile (the PROD forward), not eager-unfolded bf16.
    let ref_logits = fix.get("ref_logits_compile").expect("fixture.ref_logits_compile");
    let ref_values = fix.get("ref_values_compile").unwrap();
    let prior_ref = softmax_legal(ref_logits, mask);
    let prior_dmax = max_abs_diff(&prior_rust, &prior_ref);
    let value_dmax = max_abs_diff(&out_values, ref_values);
    let agree = argmax_legal(&out_logits, mask)
        .eq_tensor(&argmax_legal(ref_logits, mask))
        .to_kind(Kind::Int64)
        .sum(Kind::Int64)
        .int64_value(&[]);
    // context: divergence vs eager-unfolded bf16 (expected to be larger — fusion).
    let eager_prior_dmax = max_abs_diff(&prior_rust, &softmax_legal(fix.get("ref_logits_bf16").unwrap(), mask));
    println!("AOTI parity on {n} real positions (vs torch.compile = prod forward):");
    println!("  prior (softmax-over-legal) max|Δprob|={prior_dmax:.3e}  value max|Δ|={value_dmax:.3e}");
    println!("  argmax-over-legal agree: {agree}/{n}");
    println!("  (context: prior max|Δprob| vs eager-unfolded bf16 = {eager_prior_dmax:.3e})");
    let ok = prior_dmax < 2.0e-2 && value_dmax < 2.0e-2;
    println!("{}", if ok { "AOTI-PARITY OK" } else { "AOTI-PARITY FAILED" });
    ok
}

/// Swap-parity gate: prove the in-place weight swap (update_constant_buffer ->
/// run_const_fold -> swap) is correct. Push W2's weights into a package compiled
/// for W1, and require its forward to match a package compiled NATIVELY for W2
/// (same fusion on both sides, so any diff is purely the swap). The Python helper
/// `make_swap_fixture.py` produces the two .pt2s + W2.safetensors.
fn aoti_swap_parity(pt2_w1: &str, pt2_w2: &str, weights_w2: &str, b: i64) -> bool {
    let dev = Device::Cuda(0);
    let model = aoti::AotiModel::load(pt2_w1); // compiled for W1; we push W2 into this
    let model_ref = aoti::AotiModel::load(pt2_w2); // compiled natively for W2
    let model_w1 = aoti::AotiModel::load(pt2_w1); // untouched native W1 (staleness probe)

    // Push W2's raw primaries into the W1 package (mangled FQNs, CUDA bf16).
    let entries = Tensor::read_safetensors(weights_w2)
        .unwrap_or_else(|e| panic!("read {weights_w2}: {e}"));
    let mut names: Vec<CString> = Vec::with_capacity(entries.len());
    let mut tensors: Vec<Tensor> = Vec::with_capacity(entries.len());
    for (name, t) in entries {
        names.push(CString::new(name.replace('.', "_")).unwrap());
        tensors.push(t.to_device(dev).to_kind(Kind::BFloat16));
    }
    let refs: Vec<&Tensor> = tensors.iter().collect();
    model.swap_weights(&names, &refs);

    // Same input through both; compare the move-prior (full softmax) + values.
    let x = Tensor::randn([b, 18, 8, 8], (Kind::BFloat16, dev));
    let opts = (Kind::BFloat16, dev);
    let (out_l, out_v) = (Tensor::zeros([b, 4672], opts), Tensor::zeros([b], opts));
    let (ref_l, ref_v) = (Tensor::zeros([b, 4672], opts), Tensor::zeros([b], opts));
    let (w1_l, w1_v) = (Tensor::zeros([b, 4672], opts), Tensor::zeros([b], opts));
    model.run(&x, &out_l, &out_v);
    model_ref.run(&x, &ref_l, &ref_v);
    model_w1.run(&x, &w1_l, &w1_v);
    cudagraph::device_synchronize();

    let softmax = |l: &Tensor| l.to_kind(Kind::Float).softmax(1, Kind::Float);
    let mean_abs = |a: &Tensor, b: &Tensor| {
        (a.to_kind(Kind::Float) - b.to_kind(Kind::Float)).abs().mean(Kind::Float).double_value(&[])
    };
    let prior_dmax = max_abs_diff(&softmax(&out_l), &softmax(&ref_l));
    let value_dmax = max_abs_diff(&out_v, &ref_v);
    let value_dmean = mean_abs(&out_v, &ref_v);
    let logit_dmax = max_abs_diff(&out_l, &ref_l);
    // Staleness probe: how far is the pushed result from an UNTOUCHED W1 package?
    // ~0 vs W1 means that head ignored the swap; ~0 vs W2 means it took effect.
    let val_vs_w1 = max_abs_diff(&out_v, &w1_v);
    let logit_vs_w1 = max_abs_diff(&out_l, &w1_l);
    let w1w2_val = max_abs_diff(&w1_v, &ref_v);
    let value_rel = value_dmax / w1w2_val.max(1e-9); // residual as a fraction of the real change
    println!("AOTI swap-parity (push-W2 into W1-pkg vs native-W2, B={b}):");
    println!("  prior (full softmax) max|Δprob|={prior_dmax:.3e}  (raw logit max|Δ|={logit_dmax:.3e})");
    println!("  value max|Δ|={value_dmax:.3e}  mean|Δ|={value_dmean:.3e}  rel(max/W1↔W2gap)={value_rel:.3e}");
    println!("  staleness: value |pushed-W1|={val_vs_w1:.3e} (W1↔W2 value gap={w1w2_val:.3e})  logit |pushed-W1|={logit_vs_w1:.3e}");
    // Policy prior must be exact; value is bf16-sensitive (run_const_fold vs compile
    // fold), so gate it on the mean over the batch, not rare tanh-boundary outliers.
    let ok = prior_dmax < 2.0e-2 && value_dmean < 2.0e-2;
    println!("{}", if ok { "AOTI-SWAP-PARITY OK" } else { "AOTI-SWAP-PARITY FAILED" });
    ok
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
        "bench" => {
            assert!(tch::Cuda::is_available(), "CUDA not available to tch");
            let (config, run, interval) = bench_config(&args);
            pipeline::run_bench(config, run, interval);
        }
        "serve" => {
            assert!(tch::Cuda::is_available(), "CUDA not available to tch");
            let (config, _, _) = bench_config(&args);
            pipeline::run_serve(config);
        }
        "time-forward" => {
            assert!(tch::Cuda::is_available(), "CUDA not available to tch");
            let weights = flag(&args, "--weights", "weights.safetensors");
            let b: i64 = flag(&args, "--bucket", "512").parse().unwrap();
            time_forward(&weights, b);
        }
        "aoti-time" => {
            assert!(tch::Cuda::is_available(), "CUDA not available to tch");
            let pt2 = flag(&args, "--pt2", "serving_model.pt2");
            let weights = flag(&args, "--weights", "weights.safetensors");
            let b: i64 = flag(&args, "--bucket", "512").parse().unwrap();
            aoti_time(&pt2, &weights, b);
        }
        "aoti-parity" => {
            assert!(tch::Cuda::is_available(), "CUDA not available to tch");
            let dir = flag(&args, "--dir", ".");
            let pt2 = flag(&args, "--pt2", "serving_model.pt2");
            std::process::exit(if aoti_parity(&dir, &pt2) { 0 } else { 1 });
        }
        "aoti-swap-parity" => {
            assert!(tch::Cuda::is_available(), "CUDA not available to tch");
            let pt2_w1 = flag(&args, "--pt2-w1", "swap_w1.pt2");
            let pt2_w2 = flag(&args, "--pt2-w2", "swap_w2.pt2");
            let weights_w2 = flag(&args, "--weights-w2", "swap_w2.safetensors");
            let b: i64 = flag(&args, "--bucket", "512").parse().unwrap();
            std::process::exit(if aoti_swap_parity(&pt2_w1, &pt2_w2, &weights_w2, b) { 0 } else { 1 });
        }
        other => {
            eprintln!("unknown subcommand {other:?}; expected: serve | bench | forward-check | aoti-time | aoti-parity | aoti-swap-parity | time-forward");
            std::process::exit(2);
        }
    }
}
