// Single-game PUCT MCTS over a chess.js board, using the ONNX net for
// leaf evaluation. Sequential (one inference at a time) — fine for v1
// since the model is tiny and 200-400 sims fit comfortably in onnxruntime
// on most laptops.
//
// Sign convention (matches training/mcts.py):
//   - Node.q is stored from the node's side-to-move POV.
//   - During PUCT selection from a parent, we negate child.q (a high q for
//     the child is a bad outcome for the parent's side).
//   - During backprop, we flip the value sign at each ply walking upward.

import { Chess } from 'chess.js'

import { encodePosition, legalMoveLookup, moveToIndex } from './encode.js'
import { evaluate } from './net.js'

const C_PUCT = 1.5

class Node {
  constructor(prior = 0.0) {
    this.prior = prior
    this.visitCount = 0
    this.valueSum = 0.0
    this.children = [] // array of { move, node }
    this.expanded = false
  }
  get q() {
    return this.visitCount > 0 ? this.valueSum / this.visitCount : 0.0
  }
}

function selectChild(node) {
  const sqrtParent = Math.sqrt(Math.max(node.visitCount, 1))
  let best = null
  let bestScore = -Infinity
  for (const c of node.children) {
    const score = -c.node.q + C_PUCT * c.node.prior * sqrtParent / (1 + c.node.visitCount)
    if (score > bestScore) {
      bestScore = score
      best = c
    }
  }
  return best
}

function terminalValue(chess) {
  if (!chess.isGameOver()) return null
  if (chess.isCheckmate()) return -1.0 // side-to-move just got mated
  return 0.0
}

async function expand(node, chess) {
  const input = encodePosition(chess)
  const { policy: logits, value } = await evaluate(input)

  const { mask, moves } = legalMoveLookup(chess)
  // Softmax over legal moves only.
  let maxLogit = -Infinity
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] && logits[i] > maxLogit) maxLogit = logits[i]
  }
  let sumExp = 0
  const exp = new Float64Array(logits.length)
  for (let i = 0; i < logits.length; i++) {
    if (mask[i]) {
      const e = Math.exp(logits[i] - maxLogit)
      exp[i] = e
      sumExp += e
    }
  }

  node.children = moves.map((m) => {
    const idx = moveToIndex(m, chess)
    const prior = sumExp > 0 ? exp[idx] / sumExp : 0
    return { move: m, node: new Node(prior) }
  })
  node.expanded = true
  return value
}

function backprop(path, value) {
  let v = value
  for (let i = path.length - 1; i >= 0; i--) {
    path[i].visitCount += 1
    path[i].valueSum += v
    v = -v
  }
}

/** Return the bot's move for the given FEN. */
export async function chooseMove(fen, { sims = 400, timeBudgetMs = 1500 } = {}) {
  const chess = new Chess(fen)
  if (chess.isGameOver()) return null

  const root = new Node()
  await expand(root, chess)

  // Edge case: only one legal move — skip MCTS entirely.
  if (root.children.length === 1) return root.children[0].move

  const startedAt = (typeof performance !== 'undefined' ? performance.now() : Date.now())
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())

  for (let s = 0; s < sims; s++) {
    if (now() - startedAt > timeBudgetMs) break

    const path = [root]
    let node = root
    let depth = 0
    while (node.expanded && node.children.length > 0) {
      const choice = selectChild(node)
      chess.move(choice.move)
      depth += 1
      path.push(choice.node)
      node = choice.node
      if (chess.isGameOver()) break
    }

    const tval = terminalValue(chess)
    let value
    if (tval !== null) {
      value = tval
    } else {
      value = await expand(node, chess)
    }
    backprop(path, value)

    for (let i = 0; i < depth; i++) chess.undo()
  }

  // Pick the most-visited child. Ties broken by higher q.
  let best = null
  let bestVisits = -1
  let bestQ = -Infinity
  for (const c of root.children) {
    const v = c.node.visitCount
    const q = c.node.q
    if (v > bestVisits || (v === bestVisits && q > bestQ)) {
      bestVisits = v
      bestQ = q
      best = c
    }
  }
  return best.move
}
