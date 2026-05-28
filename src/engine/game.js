// Pure chess rules — a thin wrapper over chess.js. No React, no side effects
// on caller state. Positions are passed in and out as FEN strings, mirroring
// the "plain data in, plain data out" pattern of tic-tac-toe-ai-engine's
// engine/game.js.

import { Chess } from 'chess.js'

export const HUMAN = 'w'
export const BOT = 'b'

export function createGame() {
  return new Chess().fen()
}

// Returns the new FEN after applying `move`, or null if the move is illegal.
// `move` may be a SAN string (e.g. 'e4') or { from, to, promotion? }.
export function applyMove(fen, move) {
  const game = new Chess(fen)
  try {
    game.move(move)
    return game.fen()
  } catch {
    return null
  }
}

export function getTurn(fen) {
  return new Chess(fen).turn()
}

export function isGameOver(fen) {
  return new Chess(fen).isGameOver()
}

// Classifies the position. `kind` is one of:
//   'in-progress' | 'checkmate' | 'stalemate' | 'draw'
// For checkmate, `winner` is the side that delivered mate.
// For draws other than stalemate, `reason` says why.
export function getStatus(fen) {
  const game = new Chess(fen)
  if (game.isCheckmate()) {
    // The side to move is the one that got mated.
    const winner = game.turn() === HUMAN ? BOT : HUMAN
    return { kind: 'checkmate', winner }
  }
  if (game.isStalemate()) return { kind: 'stalemate' }
  if (game.isInsufficientMaterial())
    return { kind: 'draw', reason: 'insufficient material' }
  if (game.isThreefoldRepetition())
    return { kind: 'draw', reason: 'threefold repetition' }
  if (game.isDrawByFiftyMoves())
    return { kind: 'draw', reason: 'fifty-move rule' }
  if (game.isDraw()) return { kind: 'draw', reason: 'draw' }
  return { kind: 'in-progress' }
}
