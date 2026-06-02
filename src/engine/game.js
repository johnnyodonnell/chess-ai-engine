// Pure chess rules — a thin wrapper over chess.js. No React, no side effects
// beyond mutating the Chess instance the caller passes in. The instance is the
// source of truth and accumulates move history, which is what makes draw rules
// like threefold repetition work: a bare FEN loses that history.

import { Chess } from 'chess.js'

export const HUMAN = 'w'
export const BOT = 'b'

export function createGame() {
  return new Chess()
}

// Applies `move`, mutating `game`. Returns true if the move was legal (and
// applied), false otherwise. `move` may be a SAN string or { from, to, promotion? }.
export function applyMove(game, move) {
  try {
    game.move(move)
    return true
  } catch {
    return false
  }
}

// Tries a human drag as given; if it fails, retries as a queen promotion
// (covers pawn-reaches-back-rank without a picker — v1 auto-promotes to queen).
// Mutates `game`; returns true if a move was applied.
export function tryMove(game, from, to) {
  return applyMove(game, { from, to }) || applyMove(game, { from, to, promotion: 'q' })
}

export function getTurn(game) {
  return game.turn()
}

export function isGameOver(game) {
  return game.isGameOver()
}

// Classifies the position. `kind` is one of:
//   'in-progress' | 'checkmate' | 'stalemate' | 'draw'
// For checkmate, `winner` is the side that delivered mate.
// For draws other than stalemate, `reason` says why.
export function getStatus(game) {
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
