"""Board backend selection: Rust (chess_rs) or python-chess, via CHESS_BACKEND.

Both backends expose the identical interface consumed by mcts / selfplay /
evaluate, so the swap is a drop-in and instantly reversible:

    make_board()            -> a fresh start-position board
    board_from_fen(fen)     -> board from a FEN
    board.clone_search()    -> repetition-blind clone (== copy(stack=False))
    board.push(handle)      -> apply a move handle
    board.turn              -> True if white to move
    board.halfmove_clock    -> int
    board.legal_moves()     -> (handles, indices_int32_ndarray)
    board.encode()          -> (18, 8, 8) float32, oriented (== encode_position)
    board.outcome_white()   -> None ongoing | +1 white | -1 black | 0 draw
    board.terminal_value()  -> None ongoing | -1.0 stm-mated | 0.0 draw

Handles are opaque, hashable, pushable tokens (Rust: packed-u16 ints;
python-chess: chess.Move objects). `indices` are AlphaZero policy indices
parallel to `handles`, returned as an int32 ndarray for fast prior gathering.

Default backend is "python" (the reference); set CHESS_BACKEND=rust to use the
native board. Selected once at import.
"""

import os

import numpy as np

import chess

from encode import encode_position, move_to_index

BACKEND = os.environ.get("CHESS_BACKEND", "python").lower()


class PyBoard:
    """python-chess adapter — the reference backend and rollback path."""

    __slots__ = ("b",)

    def __init__(self, board=None):
        self.b = board if board is not None else chess.Board()

    @staticmethod
    def from_fen(fen):
        return PyBoard(chess.Board(fen))

    def clone_search(self):
        # stack=False => repetition-blind, exactly as the original MCTS leaf copy
        return PyBoard(self.b.copy(stack=False))

    def push(self, handle):
        self.b.push(handle)  # handle is a chess.Move

    @property
    def turn(self):
        return self.b.turn

    @property
    def halfmove_clock(self):
        return self.b.halfmove_clock

    def legal_moves(self):
        moves = list(self.b.legal_moves)
        indices = np.fromiter(
            (move_to_index(m, self.b) for m in moves),
            dtype=np.int32,
            count=len(moves),
        )
        return moves, indices

    def encode(self):
        return encode_position(self.b)

    def outcome_white(self):
        oc = self.b.outcome(claim_draw=True)
        if oc is None:
            return None
        if oc.winner is None:
            return 0
        return 1 if oc.winner == chess.WHITE else -1

    def terminal_value(self):
        oc = self.b.outcome(claim_draw=True)
        if oc is None:
            return None
        return 0.0 if oc.winner is None else -1.0


# rust_async: the native engine runs MANY MCTS worker threads (GIL released)
# feeding ONE consumer that does big batched in-process forwards — CPU/GPU
# overlap for higher games/sec (the A/B baseline). See selfplay.run_selfplay_async
# and chess_rs::AsyncMctsEngine. Shares the native `rust` board path.
USE_ASYNC_ENGINE = BACKEND == "rust_async"

# rust_pipeline: the decoupled non-blocking engine — workers never block on the
# GPU; two inference buckets feed ONE consumer, with a global ply-priority queue
# and dynamic game spawning. See selfplay.run_selfplay_pipeline and
# chess_rs::PipelineEngine. Shares the native `rust` board path.
USE_PIPELINE_ENGINE = BACKEND == "rust_pipeline"

if BACKEND in ("rust", "rust_async", "rust_pipeline"):
    import chess_rs

    _Board = chess_rs.Board

    def make_board():
        return _Board()

    def board_from_fen(fen):
        return _Board.from_fen(fen)

else:
    def make_board():
        return PyBoard()

    def board_from_fen(fen):
        return PyBoard.from_fen(fen)
