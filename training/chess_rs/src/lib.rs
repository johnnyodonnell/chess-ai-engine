//! chess_rs — pyo3 bindings over `chess_core` (pure board/encoding/MCTS).
//!
//! This cdylib is the Python-facing surface used by the parity tests
//! (`test_rust_parity.py`, `test_mcts_parity.py`) and, until the cutover, the
//! `PipelineEngine`/`AsyncMctsEngine` that called back into Python for the GPU
//! forward. The pure logic lives in `chess_core`; the 100% Rust self-play worker
//! is the separate `selfplay` binary. See `[[rust-selfplay-port]]`.

mod async_mcts;
mod mcts;
mod pipeline;

use chess_core::board::{pack_move, Board as CoreBoard};
use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArray4};
use pyo3::prelude::*;

const ENC_CH: usize = 18;
const ENC_LEN: usize = chess_core::ENC_LEN;

/// Python-facing chess board: a thin wrapper over `chess_core::Board`.
#[pyclass(name = "Board")]
pub struct PyBoard {
    pub inner: CoreBoard,
}

#[pymethods]
impl PyBoard {
    #[new]
    fn new() -> Self {
        PyBoard { inner: CoreBoard::new() }
    }

    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        let inner = CoreBoard::from_fen(fen)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyBoard { inner })
    }

    /// Mirror of `board.copy(stack=False)`: a clone that is repetition-blind.
    fn clone_search(&self) -> Self {
        PyBoard { inner: self.inner.clone_search() }
    }

    fn push(&mut self, handle: u16) {
        self.inner.push(handle);
    }

    #[getter]
    fn turn(&self) -> bool {
        self.inner.turn()
    }

    #[getter]
    fn halfmove_clock(&self) -> u32 {
        self.inner.halfmove_clock()
    }

    /// (handles u16, policy indices i32) for all legal moves, parallel arrays.
    fn legal_moves<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<u16>>, Bound<'py, PyArray1<i32>>) {
        let moves = self.inner.legal_moves_vec();
        let mut handles = Vec::with_capacity(moves.len());
        let mut indices = Vec::with_capacity(moves.len());
        for m in moves {
            handles.push(pack_move(m));
            indices.push(self.inner.move_to_index(m));
        }
        (handles.into_pyarray(py), indices.into_pyarray(py))
    }

    /// (18, 8, 8) float32 oriented tensor, identical to encode_position.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let mut buf = vec![0.0f32; ENC_LEN];
        self.inner.encode_into(&mut buf);
        Array3::from_shape_vec((ENC_CH, 8, 8), buf)
            .unwrap()
            .into_pyarray(py)
    }

    /// White-POV result: None ongoing, +1 white win, -1 black win, 0 draw.
    fn outcome_white(&self) -> Option<i8> {
        self.inner.outcome_white()
    }

    /// Side-to-move POV terminal value (for MCTS): None ongoing, 0.0 draw,
    /// -1.0 side-to-move was just mated.
    fn terminal_value(&self) -> Option<f32> {
        self.inner.terminal_value()
    }
}

/// Batch encode many boards into one (N, 18, 8, 8) array (one PyO3 call).
#[pyfunction]
fn encode_many<'py>(
    py: Python<'py>,
    boards: Vec<Bound<'py, PyBoard>>,
) -> Bound<'py, PyArray4<f32>> {
    let n = boards.len();
    let mut buf = vec![0.0f32; n * ENC_LEN];
    for (i, b) in boards.iter().enumerate() {
        let board = b.borrow();
        board.inner.encode_into(&mut buf[i * ENC_LEN..(i + 1) * ENC_LEN]);
    }
    Array4::from_shape_vec((n, ENC_CH, 8, 8), buf)
        .unwrap()
        .into_pyarray(py)
}

#[pymodule]
fn chess_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<mcts::MctsEngine>()?;
    m.add_class::<async_mcts::AsyncMctsEngine>()?;
    m.add_class::<pipeline::PipelineEngine>()?;
    m.add_function(wrap_pyfunction!(encode_many, m)?)?;
    Ok(())
}
