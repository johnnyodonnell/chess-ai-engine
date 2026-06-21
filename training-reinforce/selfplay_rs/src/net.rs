//! The 4x96 AlphaZero ResNet (training/net.py `ChessNet`) in tch-rs.
//!
//! Explicit-BatchNorm forward only (matches PyTorch bit-exactly) — this worker is
//! not throughput-critical the way the MCTS one is, so we skip the BN-folding /
//! AOTInductor / CUDA-graph machinery of training/selfplay_rs and keep a single
//! parity-exact path. Weights load from fp32 safetensors keyed by state_dict FQN
//! (stem_conv/stem_bn, tower.{i}.{conv1,bn1,conv2,bn2}, policy_conv/policy_bn/
//! policy_fc, value_conv/value_bn/value_fc1/value_fc2).

use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

const BN_EPS: f64 = 1e-5;
const BN_MOMENTUM: f64 = 0.1;

pub struct Net {
    p: HashMap<String, Tensor>,
    n_blocks: usize,
}

fn load_map(path: &str, dev: Device, kind: Kind) -> HashMap<String, Tensor> {
    Tensor::read_safetensors(path)
        .unwrap_or_else(|e| panic!("read_safetensors {path}: {e}"))
        .into_iter()
        .map(|(k, v)| {
            // BN tracks an int64 num_batches_tracked buffer; keep its dtype.
            let t = if k.ends_with("num_batches_tracked") {
                v.to_device(dev)
            } else {
                v.to_device(dev).to_kind(kind)
            };
            (k, t)
        })
        .collect()
}

impl Net {
    pub fn load(path: &str, dev: Device, kind: Kind) -> Net {
        let p = load_map(path, dev, kind);
        let n_blocks = (0..)
            .take_while(|i| p.contains_key(&format!("tower.{i}.conv1.weight")))
            .count();
        Net { p, n_blocks }
    }

    fn g(&self, k: &str) -> &Tensor {
        self.p.get(k).unwrap_or_else(|| panic!("missing param {k}"))
    }

    fn conv(&self, x: &Tensor, w: &str, pad: i64) -> Tensor {
        let none: Option<Tensor> = None;
        x.conv2d(self.g(w), none.as_ref(), [1, 1], [pad, pad], [1, 1], 1)
    }

    fn bn(&self, h: &Tensor, pfx: &str) -> Tensor {
        h.batch_norm(
            Some(self.g(&format!("{pfx}.weight"))),
            Some(self.g(&format!("{pfx}.bias"))),
            Some(self.g(&format!("{pfx}.running_mean"))),
            Some(self.g(&format!("{pfx}.running_var"))),
            false, // not training: use running stats
            BN_MOMENTUM,
            BN_EPS,
            true, // cudnn enabled
        )
    }

    /// (policy_logits [B,4672], value [B]) — matches ChessNet.forward exactly.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = self.bn(&self.conv(x, "stem_conv.weight", 1), "stem_bn").relu();
        for i in 0..self.n_blocks {
            let inp = h.shallow_clone();
            let k = self
                .bn(&self.conv(&inp, &format!("tower.{i}.conv1.weight"), 1), &format!("tower.{i}.bn1"))
                .relu();
            let k = self.bn(&self.conv(&k, &format!("tower.{i}.conv2.weight"), 1), &format!("tower.{i}.bn2"));
            h = (k + inp).relu();
        }

        let policy = self
            .bn(&self.conv(&h, "policy_conv.weight", 0), "policy_bn")
            .relu()
            .flatten(1, -1)
            .linear(self.g("policy_fc.weight"), Some(self.g("policy_fc.bias")));

        let value = self
            .bn(&self.conv(&h, "value_conv.weight", 0), "value_bn")
            .relu()
            .flatten(1, -1)
            .linear(self.g("value_fc1.weight"), Some(self.g("value_fc1.bias")))
            .relu()
            .linear(self.g("value_fc2.weight"), Some(self.g("value_fc2.bias")))
            .tanh()
            .squeeze_dim(-1);

        (policy, value)
    }
}
