//! chess_core — pure (pyo3-free) chess board, AlphaZero encoding, and MCTS
//! primitives shared by the `chess_rs` cdylib (parity tests) and the `selfplay`
//! binary (the 100% Rust self-play worker). See `[[rust-selfplay-port]]`.

pub mod board;
pub mod mcts;

pub use board::{pack_move, unpack_move, Board, ENC_LEN};
