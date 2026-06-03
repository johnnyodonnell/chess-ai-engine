//! The 4x96 AlphaZero ResNet (`net.py`) in tch-rs.
//!
//! Two forward paths:
//!   - `forward` (prod): BatchNorm **folded into the convs** at load (conv→relu
//!     tower, no BN kernels). The net is tiny-spatial (8x8) and wildly launch/
//!     memory-bound, so removing the 9 BN kernels is the main speed lever (tch has
//!     no Inductor fusion). Fold is exact in fp32: scale=gamma/sqrt(var+eps),
//!     w'=w*scale, b'=beta-mean*scale.
//!   - `forward_unfolded`: explicit BN; matches PyTorch bit-exactly. Kept as the
//!     parity reference (forward-check compares folded vs unfolded internally to
//!     isolate fold correctness from cross-framework TF32/algo noise).
//!
//! Params held as persistent tensors keyed by state_dict names; `reload` copies
//! new weights IN PLACE and re-derives the folds in place, so a captured CUDA
//! graph keeps replaying fresh weights with no recapture.

use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

const BN_EPS: f64 = 1e-5;
const BN_MOMENTUM: f64 = 0.1;

fn inplace_copy(dst: &Tensor, src: &Tensor) {
    let mut t = dst.shallow_clone();
    let _ = t.copy_(src);
}

/// A BN-folded conv: conv2d(x, w, b) with `pad`.
struct FoldedConv {
    w: Tensor,
    b: Tensor,
    pad: i64,
}
impl FoldedConv {
    fn apply(&self, x: &Tensor) -> Tensor {
        x.conv2d(&self.w, Some(&self.b), [1, 1], [self.pad, self.pad], [1, 1], 1)
    }
}

pub struct Net {
    dev: Device,
    kind: Kind,
    pub n_blocks: usize,
    p: HashMap<String, Tensor>, // raw params (unfolded path + reload source + fc)
    stem: FoldedConv,
    blocks: Vec<(FoldedConv, FoldedConv)>,
    policy_conv: FoldedConv,
    value_conv: FoldedConv,
}

fn load_map(path: &str, dev: Device, kind: Kind) -> HashMap<String, Tensor> {
    Tensor::read_safetensors(path)
        .unwrap_or_else(|e| panic!("read_safetensors {path}: {e}"))
        .into_iter()
        .map(|(k, v)| {
            let t = if k.ends_with("num_batches_tracked") {
                v.to_device(dev)
            } else {
                v.to_device(dev).to_kind(kind)
            };
            (k, t)
        })
        .collect()
}

/// Compute folded (w', b') in fp32 (precise), returned in `kind`.
fn compute_fold(p: &HashMap<String, Tensor>, conv_key: &str, bn: &str, kind: Kind) -> (Tensor, Tensor) {
    let g = |k: &str| p.get(k).unwrap_or_else(|| panic!("missing {k}")).to_kind(Kind::Float);
    let w = g(conv_key); // [out,in,kh,kw]
    let gamma = g(&format!("{bn}.weight"));
    let beta = g(&format!("{bn}.bias"));
    let mean = g(&format!("{bn}.running_mean"));
    let var = g(&format!("{bn}.running_var"));
    let out = w.size()[0];
    let scale = gamma / (var + BN_EPS).sqrt();
    let w_folded = w * scale.reshape([out, 1, 1, 1]);
    let b_folded = beta - mean * &scale;
    (w_folded.to_kind(kind), b_folded.to_kind(kind))
}

impl Net {
    pub fn load(path: &str, dev: Device, kind: Kind) -> Net {
        let p = load_map(path, dev, kind);
        let n_blocks = (0..)
            .take_while(|i| p.contains_key(&format!("tower.{i}.conv1.weight")))
            .count();
        let fc = |ck: &str, bn: &str, pad: i64| {
            let (w, b) = compute_fold(&p, ck, bn, kind);
            FoldedConv { w, b, pad }
        };
        let blocks = (0..n_blocks)
            .map(|i| {
                (
                    fc(&format!("tower.{i}.conv1.weight"), &format!("tower.{i}.bn1"), 1),
                    fc(&format!("tower.{i}.conv2.weight"), &format!("tower.{i}.bn2"), 1),
                )
            })
            .collect();
        Net {
            stem: fc("stem_conv.weight", "stem_bn", 1),
            policy_conv: fc("policy_conv.weight", "policy_bn", 0),
            value_conv: fc("value_conv.weight", "value_bn", 0),
            blocks,
            dev,
            kind,
            n_blocks,
            p,
        }
    }

    pub fn reload(&self, path: &str) {
        let m = load_map(path, self.dev, self.kind);
        for (k, dst) in &self.p {
            if let Some(src) = m.get(k) {
                inplace_copy(dst, src);
            }
        }
        // Re-derive folds in place (uses the just-updated self.p).
        let refold = |c: &FoldedConv, ck: &str, bn: &str| {
            let (w, b) = compute_fold(&self.p, ck, bn, self.kind);
            inplace_copy(&c.w, &w);
            inplace_copy(&c.b, &b);
        };
        refold(&self.stem, "stem_conv.weight", "stem_bn");
        for (i, (c1, c2)) in self.blocks.iter().enumerate() {
            refold(c1, &format!("tower.{i}.conv1.weight"), &format!("tower.{i}.bn1"));
            refold(c2, &format!("tower.{i}.conv2.weight"), &format!("tower.{i}.bn2"));
        }
        refold(&self.policy_conv, "policy_conv.weight", "policy_bn");
        refold(&self.value_conv, "value_conv.weight", "value_bn");
    }

    fn g(&self, k: &str) -> &Tensor {
        self.p.get(k).unwrap_or_else(|| panic!("missing param {k}"))
    }

    fn heads(&self, h: &Tensor, pol_pre: Tensor, val_pre: Tensor) -> (Tensor, Tensor) {
        let _ = h;
        let p = pol_pre
            .relu()
            .flatten(1, -1)
            .linear(self.g("policy_fc.weight"), Some(self.g("policy_fc.bias")));
        let v = val_pre
            .relu()
            .flatten(1, -1)
            .linear(self.g("value_fc1.weight"), Some(self.g("value_fc1.bias")))
            .relu()
            .linear(self.g("value_fc2.weight"), Some(self.g("value_fc2.bias")))
            .tanh()
            .squeeze_dim(-1);
        (p, v)
    }

    /// Prod forward. NOTE: BN-folding is bit-lossy in bf16 (baking the per-channel
    /// scale into 8-bit-mantissa weights drifts ~0.1 vs explicit BN — verified by
    /// forward-check's folded-vs-unfolded diff), so prod uses the explicit-BN path
    /// (parity-exact). The folded path/fields are retained only for the diagnostic.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        self.forward_unfolded(x)
    }

    /// BN-folded forward (DIAGNOSTIC ONLY — lossy in bf16). conv→relu, no BN.
    pub fn forward_folded(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = self.stem.apply(x).relu();
        for (c1, c2) in &self.blocks {
            let inp = h.shallow_clone();
            let k = c1.apply(&inp).relu();
            let k = c2.apply(&k);
            h = (k + inp).relu();
        }
        let pol_pre = self.policy_conv.apply(&h);
        let val_pre = self.value_conv.apply(&h);
        self.heads(&h, pol_pre, val_pre)
    }

    fn bn(&self, h: &Tensor, pfx: &str) -> Tensor {
        h.batch_norm(
            Some(self.g(&format!("{pfx}.weight"))),
            Some(self.g(&format!("{pfx}.bias"))),
            Some(self.g(&format!("{pfx}.running_mean"))),
            Some(self.g(&format!("{pfx}.running_var"))),
            false,
            BN_MOMENTUM,
            BN_EPS,
            true,
        )
    }
    fn conv(&self, x: &Tensor, w: &str, pad: i64) -> Tensor {
        let none: Option<Tensor> = None;
        x.conv2d(self.g(w), none.as_ref(), [1, 1], [pad, pad], [1, 1], 1)
    }

    /// Explicit-BN forward (parity reference; matches PyTorch bit-exactly).
    pub fn forward_unfolded(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut h = self.bn(&self.conv(x, "stem_conv.weight", 1), "stem_bn").relu();
        for i in 0..self.n_blocks {
            let inp = h.shallow_clone();
            let k = self
                .bn(&self.conv(&inp, &format!("tower.{i}.conv1.weight"), 1), &format!("tower.{i}.bn1"))
                .relu();
            let k = self.bn(&self.conv(&k, &format!("tower.{i}.conv2.weight"), 1), &format!("tower.{i}.bn2"));
            h = (k + inp).relu();
        }
        let pol_pre = self.bn(&self.conv(&h, "policy_conv.weight", 0), "policy_bn");
        let val_pre = self.bn(&self.conv(&h, "value_conv.weight", 0), "value_bn");
        self.heads(&h, pol_pre, val_pre)
    }
}
