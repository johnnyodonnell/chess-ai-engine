// Position and move encoding for the AlphaZero engine. This file is a
// byte-for-byte mirror of training/encode.py — any change here MUST be
// matched there (and vice versa) or the network's predictions become
// nonsense.
//
// Convention: positions are oriented from the side-to-move's perspective.
// When it's black's turn we vertically flip the board (rank r → 7-r) and
// swap piece colors, so the net always sees "my pieces at the bottom."
//
// Move encoding: 8×8×73 = 4672 indices.
//   planes  0..55   queen-like moves: 8 compass directions × 7 distances
//   planes 56..63   knight moves
//   planes 64..72   underpromotions: 3 file directions × 3 pieces (N,B,R)
// Queen promotions go through the queen-like planes (any queen-like pawn
// move reaching oriented rank 7 is decoded as a queen promotion).

export const INPUT_CHANNELS = 18
export const POLICY_PLANES = 73
export const POLICY_SIZE = 64 * POLICY_PLANES // 4672

export const QUEEN_DIRS = [
  [1, 0],   // N
  [1, 1],   // NE
  [0, 1],   // E
  [-1, 1],  // SE
  [-1, 0],  // S
  [-1, -1], // SW
  [0, -1],  // W
  [1, -1],  // NW
]

export const KNIGHT_DIRS = [
  [2, 1],
  [1, 2],
  [-1, 2],
  [-2, 1],
  [-2, -1],
  [-1, -2],
  [1, -2],
  [2, -1],
]

export const UNDER_PROMO_DFILES = [-1, 0, 1]
export const UNDER_PROMO_PIECES = ['n', 'b', 'r'] // chess.js lowercases

const PIECE_CHANNEL = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 }

function squareToRF(sq) {
  const file = sq.charCodeAt(0) - 97 // 'a' = 97
  const rank = parseInt(sq[1], 10) - 1
  return [rank, file]
}

function orientRF(rank, file, flip) {
  return flip ? [7 - rank, file] : [rank, file]
}

// chess.js's board() returns rows top-to-bottom (rank 8 at row 0).
// Convert (row, file) from board() into 0..7 rank (0 = rank 1).
function boardRowToRank(row) {
  return 7 - row
}

// Parse castling + en-passant + halfmove out of the FEN string directly.
// chess.js exposes some of this via getCastlingRights / fen(), but parsing
// once is simpler than threading the API.
function parseFenFields(fen) {
  const parts = fen.split(' ')
  return {
    castling: parts[2] || '-',
    ep: parts[3] || '-',
    halfmove: parseInt(parts[4], 10) || 0,
  }
}

export function encodePosition(chess) {
  const planes = new Float32Array(INPUT_CHANNELS * 64)
  const fen = chess.fen()
  const turn = chess.turn() // 'w' or 'b'
  const flip = turn === 'b'
  const { castling, ep, halfmove } = parseFenFields(fen)

  const board = chess.board()
  for (let row = 0; row < 8; row++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[row][f]
      if (!piece) continue
      const rank = boardRowToRank(row)
      const [orR, orF] = orientRF(rank, f, flip)
      const base = piece.color === turn ? 0 : 6
      const channel = base + PIECE_CHANNEL[piece.type]
      planes[channel * 64 + orR * 8 + orF] = 1.0
    }
  }

  const meK = turn === 'w' ? 'K' : 'k'
  const meQ = turn === 'w' ? 'Q' : 'q'
  const themK = turn === 'w' ? 'k' : 'K'
  const themQ = turn === 'w' ? 'q' : 'Q'
  const setPlane = (c, val) => {
    const off = c * 64
    for (let i = 0; i < 64; i++) planes[off + i] = val
  }
  if (castling.includes(meK)) setPlane(12, 1)
  if (castling.includes(meQ)) setPlane(13, 1)
  if (castling.includes(themK)) setPlane(14, 1)
  if (castling.includes(themQ)) setPlane(15, 1)

  if (ep !== '-') {
    const [epR, epF] = squareToRF(ep)
    const [orR, orF] = orientRF(epR, epF, flip)
    planes[16 * 64 + orR * 8 + orF] = 1.0
  }

  setPlane(17, Math.min(halfmove, 100) / 100.0)
  return planes
}

export function moveToIndex(move, chess) {
  const flip = chess.turn() === 'b'
  const [fR, fF] = squareToRF(move.from)
  const [tR, tF] = squareToRF(move.to)
  const [orFR, orFF] = orientRF(fR, fF, flip)
  const [orTR, orTF] = orientRF(tR, tF, flip)
  const drank = orTR - orFR
  const dfile = orTF - orFF

  let plane
  if (move.promotion && UNDER_PROMO_PIECES.includes(move.promotion)) {
    const dfIdx = UNDER_PROMO_DFILES.indexOf(dfile)
    const pieceIdx = UNDER_PROMO_PIECES.indexOf(move.promotion)
    plane = 64 + 3 * dfIdx + pieceIdx
  } else {
    const knightIdx = KNIGHT_DIRS.findIndex((d) => d[0] === drank && d[1] === dfile)
    if (knightIdx >= 0) {
      plane = 56 + knightIdx
    } else {
      const dist = Math.max(Math.abs(drank), Math.abs(dfile))
      const drU = drank === 0 ? 0 : drank / dist
      const dfU = dfile === 0 ? 0 : dfile / dist
      const dirIdx = QUEEN_DIRS.findIndex((d) => d[0] === drU && d[1] === dfU)
      plane = dirIdx * 7 + (dist - 1)
    }
  }

  return orFR * 8 * POLICY_PLANES + orFF * POLICY_PLANES + plane
}

// Build a Uint8Array(POLICY_SIZE) where mask[i]=1 iff index i is a legal
// move; also returns a parallel index→Move map for fast lookup during MCTS.
export function legalMoveLookup(chess) {
  const mask = new Uint8Array(POLICY_SIZE)
  const byIndex = new Map()
  const moves = chess.moves({ verbose: true })
  for (const m of moves) {
    const idx = moveToIndex(m, chess)
    mask[idx] = 1
    byIndex.set(idx, m)
  }
  return { mask, byIndex, moves }
}
