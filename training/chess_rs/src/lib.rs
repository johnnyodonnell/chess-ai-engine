//! chess_rs — native chess board + AlphaZero encoding for self-play.
//!
//! Mirrors `training/encode.py` byte-for-byte and python-chess's
//! `outcome(claim_draw=True)` semantics exactly (incl. its insufficient-material
//! rule and the `board.copy(stack=False)` repetition-blindness used inside MCTS
//! search). The Python side keeps the MCTS tree; this owns board + encoding +
//! move-indexing. See the plan / `[[rust-selfplay-port]]`.
//!
//! Move handles cross the boundary as a packed u16: from(0..63) | to<<6 |
//! promo<<12, where promo nibble = 0 None else (cozy Piece index + 1). Castling
//! is cozy's "king captures own rook" encoding in the handle (so `push` is
//! native), but its *policy index* uses python-chess's king-two-squares form.

use cozy_chess::{
    BitBoard, Board as CzBoard, Color, File, Move, Piece, Rank, Square,
};
use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArray4};
use pyo3::prelude::*;

const PLANES: i32 = 73; // POLICY_PLANES
const PLANE_AREA: usize = 8 * 8; // per-channel cells
const ENC_CH: usize = 18; // INPUT_CHANNELS
const ENC_LEN: usize = ENC_CH * PLANE_AREA; // 1152

// Dark squares (a1 dark, bit0 set). Used for the insufficient-material rule.
const DARK_BB: u64 = 0xAA55_AA55_AA55_AA55;

// ---------------------------------------------------------------------------
// Position key for repetition counting (mirrors python-chess _transposition_key:
// piece placement + side + castling + ep *only when legally capturable*).
// Stored exactly (no hashing) so there are zero false repetitions.
// ---------------------------------------------------------------------------
#[derive(Clone, PartialEq, Eq)]
struct PosKey {
    bbs: [u64; 8], // pawns,knights,bishops,rooks,queens,kings, white_occ, black_occ
    flags: u32,    // side(1) | castle(4) | ep_file+1(4)
}

fn orient(sq: Square, flip: bool) -> (i32, i32) {
    let r = sq.rank() as i32;
    let f = sq.file() as i32;
    (if flip { 7 - r } else { r }, f)
}

fn knight_dir_index(dr: i32, df: i32) -> Option<i32> {
    // KNIGHT_DIRS order from encode.py.
    match (dr, df) {
        (2, 1) => Some(0),
        (1, 2) => Some(1),
        (-1, 2) => Some(2),
        (-2, 1) => Some(3),
        (-2, -1) => Some(4),
        (-1, -2) => Some(5),
        (1, -2) => Some(6),
        (2, -1) => Some(7),
        _ => None,
    }
}

fn queen_dir_index(dr: i32, df: i32) -> i32 {
    // QUEEN_DIRS order from encode.py: N,NE,E,SE,S,SW,W,NW.
    match (dr, df) {
        (1, 0) => 0,
        (1, 1) => 1,
        (0, 1) => 2,
        (-1, 1) => 3,
        (-1, 0) => 4,
        (-1, -1) => 5,
        (0, -1) => 6,
        (1, -1) => 7,
        _ => unreachable!("non-queen-like direction ({dr},{df})"),
    }
}

fn piece_index(p: Piece) -> usize {
    p as usize // Pawn0 Knight1 Bishop2 Rook3 Queen4 King5
}

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Board {
    inner: CzBoard,
    // Some(history of position keys) on the GAME board (threefold/fivefold
    // active); None on search clones -> repetition draws never fire, mirroring
    // python-chess `copy(stack=False)`. Halfmove-based draws still fire either
    // way (halfmove clock lives on the board).
    history: Option<Vec<PosKey>>,
}

impl Board {
    fn us(&self) -> Color {
        self.inner.side_to_move()
    }
    fn occ(&self, c: Color) -> BitBoard {
        self.inner.colors(c)
    }
    fn legal_moves_vec(&self) -> Vec<Move> {
        let mut v = Vec::with_capacity(48);
        self.inner.generate_moves(|pm| {
            for m in pm {
                v.push(m);
            }
            false
        });
        v
    }
    fn has_any_legal(&self) -> bool {
        let mut any = false;
        self.inner.generate_moves(|_| {
            any = true;
            true // stop at first
        });
        any
    }
    fn in_check(&self) -> bool {
        !self.inner.checkers().is_empty()
    }

    /// File on which an en-passant capture is *legally* available (mirrors
    /// python-chess `has_legal_en_passant`): cozy only stores ep as a file, so
    /// confirm a legal ep capture actually exists this move.
    fn legal_ep_file(&self) -> Option<File> {
        let epf = self.inner.en_passant()?;
        let rank = if self.us() == Color::White {
            Rank::Sixth
        } else {
            Rank::Third
        };
        let target = Square::new(epf, rank);
        // A legal pawn move landing on the ep target square is the ep capture.
        let mut found = false;
        self.inner.generate_moves(|pm| {
            if pm.piece == Piece::Pawn {
                for m in pm {
                    if m.to == target {
                        found = true;
                        return true;
                    }
                }
            }
            false
        });
        if found {
            Some(epf)
        } else {
            None
        }
    }

    fn castle_bits(&self) -> u32 {
        let w = self.inner.castle_rights(Color::White);
        let b = self.inner.castle_rights(Color::Black);
        (w.short.is_some() as u32)
            | ((w.long.is_some() as u32) << 1)
            | ((b.short.is_some() as u32) << 2)
            | ((b.long.is_some() as u32) << 3)
    }

    fn pos_key(&self) -> PosKey {
        let p = |pc: Piece| self.inner.pieces(pc).0;
        let bbs = [
            p(Piece::Pawn),
            p(Piece::Knight),
            p(Piece::Bishop),
            p(Piece::Rook),
            p(Piece::Queen),
            p(Piece::King),
            self.occ(Color::White).0,
            self.occ(Color::Black).0,
        ];
        let side = (self.us() == Color::Black) as u32;
        let ep = match self.legal_ep_file() {
            Some(f) => (f as u32) + 1,
            None => 0,
        };
        let flags = side | (self.castle_bits() << 1) | (ep << 5);
        PosKey { bbs, flags }
    }

    fn rep_count(&self) -> usize {
        match &self.history {
            Some(h) => {
                let cur = self.pos_key();
                h.iter().filter(|k| **k == cur).count()
            }
            None => 1, // search clone: position seen "once" -> no repetition
        }
    }

    /// Convert a (possibly castling) cozy move to the (from, to, promo) used by
    /// python-chess's move_to_index: castling becomes the king two-square move.
    fn index_squares(&self, m: Move) -> (Square, Square, Option<Piece>) {
        let from = m.from;
        let us = self.us();
        let is_castle =
            from == self.inner.king(us) && self.inner.colored_pieces(us, Piece::Rook).has(m.to);
        if is_castle {
            let dest_file = if (m.to.file() as i32) > (from.file() as i32) {
                File::G
            } else {
                File::C
            };
            (from, Square::new(dest_file, from.rank()), None)
        } else {
            (from, m.to, m.promotion)
        }
    }

    fn move_to_index(&self, m: Move) -> i32 {
        let flip = self.us() == Color::Black;
        let (from, to, promo) = self.index_squares(m);
        let (fr, ff) = orient(from, flip);
        let (tr, tf) = orient(to, flip);
        let drank = tr - fr;
        let dfile = tf - ff;

        let plane = match promo {
            Some(p) if p == Piece::Knight || p == Piece::Bishop || p == Piece::Rook => {
                let dfile_idx = match dfile {
                    -1 => 0,
                    0 => 1,
                    1 => 2,
                    _ => unreachable!("underpromo dfile {dfile}"),
                };
                let piece_idx = piece_index(p) as i32 - 1; // N0 B1 R2
                64 + 3 * dfile_idx + piece_idx
            }
            _ => {
                if let Some(k) = knight_dir_index(drank, dfile) {
                    56 + k
                } else {
                    let dist = drank.abs().max(dfile.abs());
                    let dr_u = if drank == 0 { 0 } else { drank / dist };
                    let df_u = if dfile == 0 { 0 } else { dfile / dist };
                    queen_dir_index(dr_u, df_u) * 7 + (dist - 1)
                }
            }
        };
        fr * 8 * PLANES + ff * PLANES + plane
    }

    fn encode_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), ENC_LEN);
        out.iter_mut().for_each(|x| *x = 0.0);
        let flip = self.us() == Color::Black;
        let me = self.us();
        let them = !me;

        let set = |out: &mut [f32], ch: usize, r: i32, f: i32| {
            out[ch * PLANE_AREA + (r as usize) * 8 + (f as usize)] = 1.0;
        };

        let pieces = [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ];
        for (i, &pc) in pieces.iter().enumerate() {
            for sq in self.inner.colored_pieces(me, pc) {
                let (r, f) = orient(sq, flip);
                set(out, i, r, f);
            }
            for sq in self.inner.colored_pieces(them, pc) {
                let (r, f) = orient(sq, flip);
                set(out, i + 6, r, f);
            }
        }

        let cr_me = self.inner.castle_rights(me);
        let cr_them = self.inner.castle_rights(them);
        let fill = |out: &mut [f32], ch: usize| {
            for c in 0..PLANE_AREA {
                out[ch * PLANE_AREA + c] = 1.0;
            }
        };
        if cr_me.short.is_some() {
            fill(out, 12);
        }
        if cr_me.long.is_some() {
            fill(out, 13);
        }
        if cr_them.short.is_some() {
            fill(out, 14);
        }
        if cr_them.long.is_some() {
            fill(out, 15);
        }

        if let Some(epf) = self.legal_ep_file() {
            let rank = if me == Color::White {
                Rank::Sixth
            } else {
                Rank::Third
            };
            let (r, f) = orient(Square::new(epf, rank), flip);
            set(out, 16, r, f);
        }

        let hm = (self.inner.halfmove_clock() as f32).min(100.0) / 100.0;
        for c in 0..PLANE_AREA {
            out[17 * PLANE_AREA + c] = hm;
        }
    }

    /// python-chess `outcome(claim_draw=True)` resolved to: None ongoing,
    /// Some(true) decisive with `winner_white`, Some(false) draw.
    fn outcome(&self) -> Option<(bool, bool)> {
        // returns (decisive, winner_is_white); draw => (false, _ignored)
        let legal_empty = !self.has_any_legal();
        if legal_empty {
            if self.in_check() {
                // checkmate: side-to-move is mated, opponent wins.
                let winner_white = self.us() == Color::Black;
                return Some((true, winner_white));
            }
            // stalemate (also covers insufficient w/ no moves) -> draw
            return Some((false, false));
        }
        if self.is_insufficient_material() {
            return Some((false, false));
        }
        // 75-move (automatic): halfmove >= 150, legal moves exist.
        if self.inner.halfmove_clock() >= 150 {
            return Some((false, false));
        }
        // Fivefold repetition (automatic).
        if self.rep_count() >= 5 {
            return Some((false, false));
        }
        // Claimable: 50-move then threefold.
        if self.can_claim_fifty() {
            return Some((false, false));
        }
        if self.can_claim_threefold() {
            return Some((false, false));
        }
        None
    }

    fn is_insufficient_material(&self) -> bool {
        self.has_insufficient_material(Color::White) && self.has_insufficient_material(Color::Black)
    }

    fn has_insufficient_material(&self, color: Color) -> bool {
        let occ_c = self.occ(color);
        let pawns = self.inner.pieces(Piece::Pawn);
        let knights = self.inner.pieces(Piece::Knight);
        let bishops = self.inner.pieces(Piece::Bishop);
        let rooks = self.inner.pieces(Piece::Rook);
        let queens = self.inner.pieces(Piece::Queen);

        if !(occ_c & (pawns | rooks | queens)).is_empty() {
            return false;
        }
        if !(occ_c & knights).is_empty() {
            let other = self.occ(!color);
            let kings = self.inner.pieces(Piece::King);
            let selfmate_pieces = other & !kings & !queens;
            return occ_c.len() <= 2 && selfmate_pieces.is_empty();
        }
        if !(occ_c & bishops).is_empty() {
            let dark = BitBoard(DARK_BB);
            let light = BitBoard(!DARK_BB);
            let same_color = (bishops & dark).is_empty() || (bishops & light).is_empty();
            return same_color && pawns.is_empty() && knights.is_empty();
        }
        true
    }

    fn is_zeroing(&self, m: Move) -> bool {
        // pawn move or any capture (incl. ep) resets the halfmove clock.
        if self.inner.piece_on(m.from) == Some(Piece::Pawn) {
            return true;
        }
        self.occ(!self.us()).has(m.to)
    }

    fn can_claim_fifty(&self) -> bool {
        let hm = self.inner.halfmove_clock();
        if hm >= 100 {
            return true; // legal moves exist (checked before calling)
        }
        if hm >= 99 {
            for m in self.legal_moves_vec() {
                if !self.is_zeroing(m) {
                    let mut nb = self.inner.clone();
                    nb.play_unchecked(m);
                    // new halfmove == 100; needs any legal move at new pos
                    let mut any = false;
                    nb.generate_moves(|_| {
                        any = true;
                        true
                    });
                    if any {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn can_claim_threefold(&self) -> bool {
        if self.rep_count() >= 3 {
            return true;
        }
        // A legal move reaching a position already seen >= 2 times.
        let hist = match &self.history {
            Some(h) => h,
            None => return false, // search clone: no history
        };
        for m in self.legal_moves_vec() {
            let mut nb = self.inner.clone();
            nb.play_unchecked(m);
            let child = Board {
                inner: nb,
                history: None,
            };
            let key = child.pos_key();
            let seen = hist.iter().filter(|k| **k == key).count();
            if seen >= 2 {
                return true;
            }
        }
        false
    }
}

#[pymethods]
impl Board {
    #[new]
    fn new() -> Self {
        let inner = CzBoard::default();
        let b = Board {
            inner,
            history: Some(Vec::new()),
        };
        let key = b.pos_key();
        Board {
            inner: b.inner,
            history: Some(vec![key]),
        }
    }

    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        let inner = CzBoard::from_fen(fen, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        let b = Board {
            inner,
            history: Some(Vec::new()),
        };
        let key = b.pos_key();
        Ok(Board {
            inner: b.inner,
            history: Some(vec![key]),
        })
    }

    /// Mirror of `board.copy(stack=False)`: a clone that is repetition-blind.
    fn clone_search(&self) -> Self {
        Board {
            inner: self.inner.clone(),
            history: None,
        }
    }

    fn push(&mut self, handle: u16) {
        let m = unpack_move(handle);
        self.inner.play_unchecked(m);
        if let Some(h) = self.history.as_mut() {
            // recompute key after the move (borrow dance: build a temp view)
            let key = Board {
                inner: self.inner.clone(),
                history: None,
            }
            .pos_key();
            h.push(key);
        }
    }

    #[getter]
    fn turn(&self) -> bool {
        self.us() == Color::White
    }

    #[getter]
    fn halfmove_clock(&self) -> u32 {
        self.inner.halfmove_clock() as u32
    }

    /// (handles u16, policy indices i32) for all legal moves, parallel arrays.
    fn legal_moves<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<u16>>, Bound<'py, PyArray1<i32>>) {
        let moves = self.legal_moves_vec();
        let mut handles = Vec::with_capacity(moves.len());
        let mut indices = Vec::with_capacity(moves.len());
        for m in moves {
            handles.push(pack_move(m));
            indices.push(self.move_to_index(m));
        }
        (handles.into_pyarray(py), indices.into_pyarray(py))
    }

    /// (18, 8, 8) float32 oriented tensor, identical to encode_position.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let mut buf = vec![0.0f32; ENC_LEN];
        self.encode_into(&mut buf);
        Array3::from_shape_vec((ENC_CH, 8, 8), buf)
            .unwrap()
            .into_pyarray(py)
    }

    /// White-POV result: None ongoing, +1 white win, -1 black win, 0 draw.
    fn outcome_white(&self) -> Option<i8> {
        match self.outcome() {
            None => None,
            Some((false, _)) => Some(0),
            Some((true, true)) => Some(1),
            Some((true, false)) => Some(-1),
        }
    }

    /// Side-to-move POV terminal value (for MCTS): None ongoing, 0.0 draw,
    /// -1.0 side-to-move was just mated.
    fn terminal_value(&self) -> Option<f32> {
        match self.outcome() {
            None => None,
            Some((false, _)) => Some(0.0),
            Some((true, _)) => Some(-1.0), // decisive => side-to-move is mated
        }
    }
}

/// Batch encode many boards into one (N, 18, 8, 8) array (one PyO3 call).
#[pyfunction]
fn encode_many<'py>(
    py: Python<'py>,
    boards: Vec<Bound<'py, Board>>,
) -> Bound<'py, PyArray4<f32>> {
    let n = boards.len();
    let mut buf = vec![0.0f32; n * ENC_LEN];
    for (i, b) in boards.iter().enumerate() {
        let board = b.borrow();
        board.encode_into(&mut buf[i * ENC_LEN..(i + 1) * ENC_LEN]);
    }
    Array4::from_shape_vec((n, ENC_CH, 8, 8), buf)
        .unwrap()
        .into_pyarray(py)
}

fn pack_move(m: Move) -> u16 {
    let from = m.from as u16;
    let to = (m.to as u16) << 6;
    let promo = match m.promotion {
        None => 0u16,
        Some(p) => (piece_index(p) as u16) + 1,
    } << 12;
    from | to | promo
}

fn unpack_move(h: u16) -> Move {
    let from = Square::index((h & 0x3F) as usize);
    let to = Square::index(((h >> 6) & 0x3F) as usize);
    let promo = match (h >> 12) & 0xF {
        0 => None,
        n => Some(PIECES_BY_INDEX[(n - 1) as usize]),
    };
    Move {
        from,
        to,
        promotion: promo,
    }
}

const PIECES_BY_INDEX: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[pymodule]
fn chess_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_function(wrap_pyfunction!(encode_many, m)?)?;
    Ok(())
}
