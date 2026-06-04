//! selfplay_rs library surface.
//!
//! Holds the modules shared between the two binaries:
//!   - `selfplay_rs` (src/main.rs): the self-play worker + parity/bench tools.
//!   - `evaluate_rs` (src/bin/evaluate_rs.rs): the Rust port of `evaluate.py`.
//!
//! `net` is the eager tch ChessNet (BN-folded prod forward + parity-exact
//! unfolded forward); `eval` is the self-play evaluation / Elo / pool-serving
//! logic. The AOTI/CUDA-graph/pipeline machinery used only by the self-play
//! worker stays bin-local to `main.rs`.

pub mod eval;
pub mod net;
