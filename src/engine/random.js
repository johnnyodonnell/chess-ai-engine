// The v1 AI engine: pick a uniform-random legal move.
//
// Contract — same shape as tic-tac-toe-ai-engine's engines and
// fox-lite-ai-engine/src/engine/random.js:
//   bestMove(fen) -> a move object { from, to, promotion? }
// Caller must only invoke when it's the bot's turn and the game isn't over.

import { Chess } from 'chess.js'

export function bestMove(fen) {
  const moves = new Chess(fen).moves({ verbose: true })
  return moves[Math.floor(Math.random() * moves.length)]
}
