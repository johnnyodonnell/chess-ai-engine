import { useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import { bestMove } from './engine/random.js'
import {
  HUMAN,
  applyMove,
  createGame,
  getStatus,
  isGameOver,
} from './engine/game.js'

function statusMessage(fen) {
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

  function handleDrop({ from, to }) {
    if (isGameOver(fen)) return false

    const afterHuman = tryMove(fen, from, to)
    if (afterHuman === null) return false

    if (isGameOver(afterHuman)) {
      setFen(afterHuman)
      return true
    }

    const botMove = bestMove(afterHuman)
    setFen(applyMove(afterHuman, botMove))
    return true
  }

  function newGame() {
    setFen(createGame())
  }

  return (
    <main className="app">
      <h1>Chess</h1>
      <Status message={statusMessage(fen)} />
      <Board fen={fen} onDrop={handleDrop} />
      <button className="new-game" onClick={newGame}>
        New Game
      </button>
    </main>
  )
}
