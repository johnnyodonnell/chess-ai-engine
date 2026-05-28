// The AlphaZero engine, drop-in replacement for ../random.js.
//
//   await bestMove(fen)  // -> { from, to, promotion? }  OR  null if game over
//
// Async because the first call may need to fetch the ONNX model and warm up
// the inference session. Subsequent calls reuse the cached session.

import { chooseMove } from './mcts.js'
import { loadModel } from './net.js'

const SIMS = 400
const TIME_BUDGET_MS = 1500

export async function bestMove(fen) {
  return chooseMove(fen, { sims: SIMS, timeBudgetMs: TIME_BUDGET_MS })
}

// Optional: pre-warm the model so the first move doesn't pay the load cost.
export function preload() {
  return loadModel()
}
