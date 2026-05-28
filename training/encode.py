"""Position and move encoding for AlphaZero chess.

Positions are oriented from the side-to-move's perspective: when it's
black's turn we vertically flip the board (rank r → 7-r) and swap piece
colors, so the net always sees "my pieces at the bottom, moving toward
higher ranks."

Move encoding follows AlphaZero (Silver et al. 2017): 8×8×73 = 4672
indices. For a move from (from_rank, from_file) in the oriented frame,
the 73 planes are:

  planes  0..55   queen-like moves: 8 compass directions × 7 distances
  planes 56..63   knight moves
  planes 64..72   underpromotions: 3 file directions × 3 pieces (N,B,R)

Queen promotions are encoded as ordinary queen-like moves (the pawn just
happens to be reaching rank 7 in the oriented frame); the decoder treats
any queen-like pawn move to oriented rank 7 as a queen promotion.

The JS encoder in src/engine/alphazero/encode.js MUST stay byte-for-byte
identical to this file. Any change here requires a matching change there.
"""

import chess
import numpy as np

INPUT_CHANNELS = 18
POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES  # 4672

# 8 queen-like compass directions, in (drank, dfile) order.
QUEEN_DIRS = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

# 8 knight move offsets, in (drank, dfile) order.
KNIGHT_DIRS = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]

# Underpromotion file deltas (capture-left, push, capture-right) and piece
# types. Queen promotions go through QUEEN_DIRS instead.
UNDER_PROMO_DFILES = [-1, 0, 1]
UNDER_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _oriented_rf(square, flip):
    r = chess.square_rank(square)
    f = chess.square_file(square)
    if flip:
        r = 7 - r
    return r, f


def _from_oriented_rf(r, f, flip):
    if flip:
        r = 7 - r
    return chess.square(f, r)


def encode_position(board):
    """Return an (18, 8, 8) float32 tensor in the oriented frame."""
    planes = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK
    me = board.turn
    them = not me

    for piece_type in range(1, 7):  # PAWN..KING
        for sq in board.pieces(piece_type, me):
            r, f = _oriented_rf(sq, flip)
            planes[piece_type - 1, r, f] = 1.0
        for sq in board.pieces(piece_type, them):
            r, f = _oriented_rf(sq, flip)
            planes[piece_type - 1 + 6, r, f] = 1.0

    if board.has_kingside_castling_rights(me):
        planes[12].fill(1.0)
    if board.has_queenside_castling_rights(me):
        planes[13].fill(1.0)
    if board.has_kingside_castling_rights(them):
        planes[14].fill(1.0)
    if board.has_queenside_castling_rights(them):
        planes[15].fill(1.0)

    # Only set the en-passant plane when an actual capture is possible.
    # chess.js normalizes its FEN output the same way (strips "phantom" ep
    # targets), so without this guard the JS encoder would diverge from
    # the Python encoder for positions like the one right after 1. e4.
    if board.ep_square is not None and board.has_legal_en_passant():
        r, f = _oriented_rf(board.ep_square, flip)
        planes[16, r, f] = 1.0

    planes[17].fill(min(board.halfmove_clock, 100) / 100.0)

    return planes


def move_to_index(move, board):
    """Map a legal chess.Move in `board` to a policy index in [0, 4672)."""
    flip = board.turn == chess.BLACK
    fr, ff = _oriented_rf(move.from_square, flip)
    tr, tf = _oriented_rf(move.to_square, flip)
    drank = tr - fr
    dfile = tf - ff

    if move.promotion in UNDER_PROMO_PIECES:
        dfile_idx = UNDER_PROMO_DFILES.index(dfile)
        piece_idx = UNDER_PROMO_PIECES.index(move.promotion)
        plane = 64 + 3 * dfile_idx + piece_idx
    elif (drank, dfile) in KNIGHT_DIRS:
        plane = 56 + KNIGHT_DIRS.index((drank, dfile))
    else:
        dist = max(abs(drank), abs(dfile))
        dr_unit = 0 if drank == 0 else drank // dist
        df_unit = 0 if dfile == 0 else dfile // dist
        dir_idx = QUEEN_DIRS.index((dr_unit, df_unit))
        plane = dir_idx * 7 + (dist - 1)

    return fr * 8 * POLICY_PLANES + ff * POLICY_PLANES + plane


def index_to_move(idx, board):
    """Reverse of move_to_index. Returns a chess.Move in the unoriented
    frame, or None if the index doesn't decode to a board move."""
    flip = board.turn == chess.BLACK
    fr = idx // (8 * POLICY_PLANES)
    ff = (idx // POLICY_PLANES) % 8
    plane = idx % POLICY_PLANES
    promotion = None

    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr_unit, df_unit = QUEEN_DIRS[dir_idx]
        tr = fr + dr_unit * dist
        tf = ff + df_unit * dist
        from_sq = _from_oriented_rf(fr, ff, flip)
        piece = board.piece_at(from_sq)
        if tr == 7 and piece is not None and piece.piece_type == chess.PAWN:
            promotion = chess.QUEEN
    elif plane < 64:
        drank, dfile = KNIGHT_DIRS[plane - 56]
        tr = fr + drank
        tf = ff + dfile
    else:
        u = plane - 64
        dfile_idx = u // 3
        piece_idx = u % 3
        tr = fr + 1
        tf = ff + UNDER_PROMO_DFILES[dfile_idx]
        promotion = UNDER_PROMO_PIECES[piece_idx]

    if not (0 <= tr < 8 and 0 <= tf < 8):
        return None
    from_sq = _from_oriented_rf(fr, ff, flip)
    to_sq = _from_oriented_rf(tr, tf, flip)
    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_move_mask(board):
    """Return a (4672,) bool mask of legal moves at this position."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
