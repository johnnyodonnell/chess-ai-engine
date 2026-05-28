import { useEffect, useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import { bestMove, preload } from './engine/alphazero/index.js'
import {
  HUMAN,
  applyMove,
  createGame,
  getStatus,
  isGameOver,
} from './engine/game.js'

function statusMessage(fen, thinking) {
  if (thinking) return 'Bot thinking…'
  const status = getStatus(fen)
  if (status.kind === 'checkmate') {
    return status.winner === HUMAN ? 'Checkmate — you win' : 'Checkmate — bot wins'
  }
  if (status.kind === 'stalemate') return 'Stalemate'
  if (status.kind === 'draw') return `Draw — ${status.reason}`
  return 'Your turn'
}

// Try the move as given; if it fails, retry as a queen promotion (covers
// pawn-reaches-back-rank without a picker — v1 auto-promotes to queen).
function tryMove(fen, from, to) {
  const direct = applyMove(fen, { from, to })
  if (direct !== null) return direct
  return applyMove(fen, { from, to, promotion: 'q' })
}

export default function App() {
  const [fen, setFen] = useState(createGame)
  const [thinking, setThinking] = useState(false)

  // Warm up the ONNX session as soon as the page loads — saves a noticeable
  // delay on the first bot move.
  useEffect(() => {
    preload().catch((err) => console.warn('AlphaZero preload failed', err))
  }, [])

  async function handleDrop({ from, to }) {
    if (isGameOver(fen) || thinking) return false

    const afterHuman = tryMove(fen, from, to)
    if (afterHuman === null) return false

    setFen(afterHuman)
    if (isGameOver(afterHuman)) return true

    setThinking(true)
    try {
      const botMove = await bestMove(afterHuman)
      if (botMove) setFen(applyMove(afterHuman, botMove))
    } finally {
      setThinking(false)
    }
    return true
  }

  function newGame() {
    if (thinking) return
    setFen(createGame())
  }

  return (
    <main className="app">
      <h1>Chess</h1>
      <Status message={statusMessage(fen, thinking)} />
      <Board fen={fen} onDrop={handleDrop} />
      <button className="new-game" onClick={newGame} disabled={thinking}>
        New Game
      </button>
    </main>
  )
}
