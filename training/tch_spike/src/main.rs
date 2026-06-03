// Stage-1 spike. Go/no-go for the Rust self-play port:
//   1. tch 0.23 links + runs against the box's libtorch 2.12-dev (SM121).
//   2. the 4x96 ChessNet forward runs ON-GPU in bf16 (not a silent CPU fallback).
//   3. CUDA-graph capture + replay reproduces the eager forward.
// Also sanity-checks fp32 parity vs a PyTorch reference (fixture.safetensors).
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

use tch::{Device, Kind, Tensor};

extern "C" {
    fn spike_graph_new() -> *mut c_void;
    fn spike_graph_capture_begin(g: *mut c_void);
    fn spike_graph_capture_end(g: *mut c_void);
    fn spike_graph_replay(g: *mut c_void);
    fn spike_set_side_stream();
    fn spike_device_synchronize();
}

struct Net {
    p: HashMap<String, Tensor>,
    n_blocks: usize,
}

impl Net {
    /// Load named tensors from a safetensors file onto `dev` in `kind`.
    fn load(path: &str, dev: Device, kind: Kind) -> Net {
        let named = Tensor::read_safetensors(path).expect("read_safetensors");
        let mut p = HashMap::new();
        for (k, v) in named {
            // BN running stats / params cast to the run kind; num_batches_tracked
            // (int64 scalar) is unused by inference, skip casting it.
            let t = if k.ends_with("num_batches_tracked") {
                v.to_device(dev)
            } else {
                v.to_device(dev).to_kind(kind)
            };
            p.insert(k, t);
        }
        let n_blocks = (0..)
            .take_while(|i| p.contains_key(&format!("tower.{i}.conv1.weight")))
            .count();
        let nf = p.get("stem_conv.weight").unwrap().size()[0];
        println!("loaded net: n_blocks={n_blocks} n_filters={nf} ({} tensors)", p.len());
        Net { p, n_blocks }
    }

    fn g(&self, k: &str) -> &Tensor {
        self.p.get(k).unwrap_or_else(|| panic!("missing param {k}"))
    }

    fn bn(&self, h: &Tensor, pfx: &str) -> Tensor {
        h.batch_norm(
            Some(self.g(&format!("{pfx}.weight"))),
            Some(self.g(&format!("{pfx}.bias"))),
            Some(self.g(&format!("{pfx}.running_mean"))),
            Some(self.g(&format!("{pfx}.running_var"))),
            false, // training
            0.1,   // momentum (unused in eval)
            1e-5,  // eps — matches nn.BatchNorm2d default
            true,  // cudnn_enabled
        )
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let none: Option<Tensor> = None;
        // stem
        let mut h = x.conv2d(self.g("stem_conv.weight"), none.as_ref(), [1, 1], [1, 1], [1, 1], 1);
        h = self.bn(&h, "stem_bn").relu();
        // tower
        for i in 0..self.n_blocks {
            let inp = h.shallow_clone();
            let mut k = inp.conv2d(self.g(&format!("tower.{i}.conv1.weight")), none.as_ref(), [1, 1], [1, 1], [1, 1], 1);
            k = self.bn(&k, &format!("tower.{i}.bn1")).relu();
            k = k.conv2d(self.g(&format!("tower.{i}.conv2.weight")), none.as_ref(), [1, 1], [1, 1], [1, 1], 1);
            k = self.bn(&k, &format!("tower.{i}.bn2"));
            h = (k + inp).relu();
        }
        // policy head
        let mut pol = h.conv2d(self.g("policy_conv.weight"), none.as_ref(), [1, 1], [0, 0], [1, 1], 1);
        pol = self.bn(&pol, "policy_bn").relu().flatten(1, -1);
        pol = pol.linear(self.g("policy_fc.weight"), Some(self.g("policy_fc.bias")));
        // value head
        let mut val = h.conv2d(self.g("value_conv.weight"), none.as_ref(), [1, 1], [0, 0], [1, 1], 1);
        val = self.bn(&val, "value_bn").relu().flatten(1, -1);
        val = val.linear(self.g("value_fc1.weight"), Some(self.g("value_fc1.bias"))).relu();
        val = val
            .linear(self.g("value_fc2.weight"), Some(self.g("value_fc2.bias")))
            .tanh()
            .squeeze_dim(-1);
        (pol, val)
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.to_kind(Kind::Float).to_device(Device::Cpu);
    let b = b.to_kind(Kind::Float).to_device(Device::Cpu);
    (a - b).abs().max().double_value(&[])
}

fn main() {
    assert!(tch::Cuda::is_available(), "CUDA not available to tch");
    let dev = Device::Cuda(0);
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let wpath = format!("{dir}/weights.safetensors");
    let fpath = format!("{dir}/fixture.safetensors");

    // Fixture: input + PyTorch fp32 reference outputs.
    let fix: HashMap<String, Tensor> = Tensor::read_safetensors(&fpath)
        .expect("read fixture")
        .into_iter()
        .collect();
    let x_cpu = fix.get("input").unwrap();
    let ref_logits = fix.get("ref_logits").unwrap();
    let ref_values = fix.get("ref_values").unwrap();
    let b = x_cpu.size()[0];
    println!("fixture: batch={b}");

    // ---- (A) fp32 parity vs PyTorch ----
    {
        let net = Net::load(&wpath, dev, Kind::Float);
        let x = x_cpu.to_device(dev).to_kind(Kind::Float);
        let (pl, vl) = net.forward(&x);
        println!(
            "fp32 GPU forward: logits.device={:?} max|Δlogits|={:.2e} max|Δvalues|={:.2e}",
            pl.device(),
            max_abs_diff(&pl, ref_logits),
            max_abs_diff(&vl, ref_values),
        );
    }

    // ---- (B) bf16 ON-GPU + throughput (detect silent CPU fallback) ----
    let net = Net::load(&wpath, dev, Kind::BFloat16);
    let x = x_cpu.to_device(dev).to_kind(Kind::BFloat16);
    let (pl_eager, _vl_eager) = net.forward(&x);
    assert!(matches!(pl_eager.device(), Device::Cuda(_)), "bf16 output not on CUDA!");
    println!(
        "bf16 GPU forward: logits.device={:?} max|Δlogits vs fp32 ref|={:.2e}",
        pl_eager.device(),
        max_abs_diff(&pl_eager, ref_logits),
    );
    unsafe { spike_device_synchronize() };
    let iters = 1000;
    let t = Instant::now();
    let mut last = (Tensor::new(), Tensor::new());
    for _ in 0..iters {
        last = net.forward(&x);
    }
    unsafe { spike_device_synchronize() };
    let dt = t.elapsed().as_secs_f64();
    println!(
        "bf16 eager throughput: {:.0} forwards/sec ({:.0} us/forward) — CPU fallback would be ~100x slower",
        iters as f64 / dt,
        dt / iters as f64 * 1e6,
    );
    drop(last);

    // ---- (C) CUDA graph capture + replay reproduces eager ----
    unsafe { spike_set_side_stream() };
    // warmup on the side stream
    for _ in 0..3 {
        let _ = net.forward(&x);
    }
    unsafe { spike_device_synchronize() };
    let graph = unsafe { spike_graph_new() };
    unsafe { spike_graph_capture_begin(graph) };
    let (mut pl_static, mut vl_static) = net.forward(&x); // captured outputs (static storage)
    unsafe { spike_graph_capture_end(graph) };
    unsafe { spike_device_synchronize() };

    // Zero the captured outputs, replay, confirm they get re-filled to match eager.
    let _ = pl_static.zero_();
    let _ = vl_static.zero_();
    unsafe { spike_graph_replay(graph) };
    unsafe { spike_device_synchronize() };
    println!(
        "graph replay vs eager: max|Δlogits|={:.2e} max|Δvalues|={:.2e}",
        max_abs_diff(&pl_static, &pl_eager),
        max_abs_diff(&vl_static, &_vl_eager),
    );

    // graph replay throughput
    unsafe { spike_device_synchronize() };
    let t = Instant::now();
    for _ in 0..iters {
        unsafe { spike_graph_replay(graph) };
    }
    unsafe { spike_device_synchronize() };
    let dt = t.elapsed().as_secs_f64();
    println!(
        "bf16 graph-replay throughput: {:.0} replays/sec ({:.0} us/replay)",
        iters as f64 / dt,
        dt / iters as f64 * 1e6,
    );

    println!("SPIKE OK");
}
