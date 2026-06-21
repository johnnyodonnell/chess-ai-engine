//! selfplay_reinforce — search-free REINFORCE self-play worker for chess-ai-engine.
//!
//! Plays full games current-vs-current with the loaded net, sampling each move
//! from the temperature-scaled, legal-masked policy (no MCTS). Every decision is
//! recorded and rewarded by the terminal game outcome (from the mover's POV) into
//! a flat f32 cohort the Python REINFORCE trainer consumes.

pub mod net;
pub mod pipeline;
pub mod selfplay;
