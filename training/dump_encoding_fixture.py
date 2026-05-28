"""Dump per-FEN encoding facts for the JS parity test.

Writes JSON to stdout. Each entry has:
  - fen: starting FEN
  - channel_sums: total per-channel sum of the encoded tensor (length 18)
  - probes: a few (channel, oriented_rank, oriented_file, value) probes
            that should match exactly
  - legal_indices: sorted list of policy indices that are legal
  - move_index_pairs: array of [{from, to, promotion?}, expected_index]
"""

import json
import sys

import chess

from encode import (
    INPUT_CHANNELS,
    encode_position,
    legal_move_mask,
    move_to_index,
)


FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r3k2r/pppqbppp/2np1n2/4p3/4P3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 0 1",
    # Pawn about to promote (white).
    "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
    # En-passant available.
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
]


def dump_one(fen):
    board = chess.Board(fen)
    tensor = encode_position(board)

    channel_sums = [float(tensor[c].sum()) for c in range(INPUT_CHANNELS)]

    probes = []
    # For each channel, find one nonzero cell to probe.
    for c in range(INPUT_CHANNELS):
        nz = tensor[c].nonzero()
        if nz[0].size > 0:
            r, f = int(nz[0][0]), int(nz[1][0])
            probes.append({"c": c, "r": r, "f": f, "v": float(tensor[c, r, f])})

    legal_mask = legal_move_mask(board)
    legal_indices = [int(i) for i in legal_mask.nonzero()[0]]

    move_index_pairs = []
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        m = {"from": chess.square_name(move.from_square), "to": chess.square_name(move.to_square)}
        if move.promotion:
            m["promotion"] = chess.piece_symbol(move.promotion)
        move_index_pairs.append([m, int(idx)])

    return {
        "fen": fen,
        "channel_sums": channel_sums,
        "probes": probes,
        "legal_indices": legal_indices,
        "move_index_pairs": move_index_pairs,
    }


if __name__ == "__main__":
    out = [dump_one(fen) for fen in FENS]
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
