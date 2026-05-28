"""Sanity tests for encode.py. Run as: python -m unittest training/test_encode.py

These same fixtures (FENs, expected sums, move↔index mappings) are mirrored
in the JS encoder tests to keep the two implementations in lockstep."""

import unittest

import chess
import numpy as np

from encode import (
    INPUT_CHANNELS,
    POLICY_SIZE,
    encode_position,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


FIXTURES = [
    chess.STARTING_FEN,
    # After 1. e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # After 1. e4 e5 2. Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    # Endgame: K+Q vs K
    "8/8/8/8/4k3/8/4Q3/4K3 w - - 0 1",
    # Castling-rich middlegame
    "r3k2r/pppqbppp/2np1n2/4p3/4P3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 0 1",
]


class TestEncodePosition(unittest.TestCase):
    def test_shape_and_dtype(self):
        board = chess.Board()
        x = encode_position(board)
        self.assertEqual(x.shape, (INPUT_CHANNELS, 8, 8))
        self.assertEqual(x.dtype, np.float32)

    def test_starting_position_orientation(self):
        # White to move: my pieces (channels 0..5) should sum to 16, same for opponent.
        x = encode_position(chess.Board())
        self.assertEqual(x[0:6].sum(), 16)
        self.assertEqual(x[6:12].sum(), 16)
        # My pawn rank should be rank 1 (white's 2nd rank from white's POV).
        self.assertTrue(np.array_equal(x[0, 1, :], np.ones(8)))

    def test_black_to_move_orientation(self):
        # After 1.e4: black to move. From black's POV, their pawns should be at
        # rank 1 of the *oriented* board (i.e., flipped).
        b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        x = encode_position(b)
        # Black pawns originally on rank 6 → oriented rank 1.
        self.assertTrue(np.array_equal(x[0, 1, :], np.ones(8)))

    def test_castling_planes(self):
        x = encode_position(chess.Board())
        self.assertEqual(x[12].sum(), 64)  # my kingside
        self.assertEqual(x[13].sum(), 64)  # my queenside
        self.assertEqual(x[14].sum(), 64)  # opp kingside
        self.assertEqual(x[15].sum(), 64)  # opp queenside

    def test_en_passant_plane(self):
        # We only set the ep plane when an actual ep capture is legal — this
        # matches chess.js's FEN normalization. After 1.e4 c6 2.e5 d5, white
        # has a real exd6 ep capture available, so the plane is set.
        b = chess.Board("rnbqkbnr/pp2pppp/2p5/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        x = encode_position(b)
        self.assertEqual(x[16].sum(), 1)
        # d6 = file 3, rank 5. White to move ⇒ no flip ⇒ (5, 3).
        self.assertEqual(x[16, 5, 3], 1.0)

    def test_en_passant_phantom_dropped(self):
        # After 1.e4 (black to move) the FEN names e3 as the ep target, but
        # no black pawn is adjacent to e4 yet, so the ep plane stays empty.
        b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        x = encode_position(b)
        self.assertEqual(x[16].sum(), 0)


class TestMoveIndex(unittest.TestCase):
    def test_round_trip_all_legal(self):
        for fen in FIXTURES:
            board = chess.Board(fen)
            for move in board.legal_moves:
                idx = move_to_index(move, board)
                self.assertTrue(0 <= idx < POLICY_SIZE, f"idx {idx} out of range for {move}")
                back = index_to_move(idx, board)
                self.assertEqual(back, move, f"FEN {fen}: {move} ↔ idx {idx} ↔ {back}")

    def test_legal_mask_count_matches(self):
        for fen in FIXTURES:
            board = chess.Board(fen)
            mask = legal_move_mask(board)
            self.assertEqual(int(mask.sum()), board.legal_moves.count())

    def test_underpromotion_planes(self):
        # Pawn on a7, can promote to a8 (push). Underpromos go to planes 64..72.
        b = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
        moves_by_promo = {m.promotion: m for m in b.legal_moves if m.from_square == chess.A7}
        # Queen push: queen-like plane.
        q_idx = move_to_index(moves_by_promo[chess.QUEEN], b)
        q_plane = q_idx % 73
        self.assertLess(q_plane, 56)
        # Underpromos: planes 64..72.
        for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            idx = move_to_index(moves_by_promo[p], b)
            plane = idx % 73
            self.assertTrue(64 <= plane < 73, f"underpromo plane {plane} for {p}")


if __name__ == "__main__":
    unittest.main()
