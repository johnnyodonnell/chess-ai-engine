import { useEffect, useRef, useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import { bestMove, preload } from './engine/alphazero/index.js'
import {
  HUMAN,
  applyMove,
  createGame,
  getStatus,
  getTurn,
  isGameOver,
  tryMove,
} from './engine/game.js'

function statusMessage(game, thinking) {
  if (thinking) return 'Bot thinking…'
  const status = getStatus(game)
  if (status.kind === 'checkmate') {
    return status.winner === HUMAN ? 'Checkmate — you win' : 'Checkmate — bot wins'
  }
  if (status.kind === 'stalemate') return 'Stalemate'
  if (status.kind === 'draw') return `Draw — ${status.reason}`
  return 'Your turn'
}

export default function App() {
  // The Chess instance is the source of truth — held in a ref so its move
  // history (and thus repetition/fifty-move detection) survives across renders.
  // `fen` exists only to drive rendering; we bump it after each mutation.
  const gameRef = useRef(createGame())
  const [fen, setFen] = useState(() => gameRef.current.fen())
  const [thinking, setThinking] = useState(false)

  // Warm up the ONNX session as soon as the page loads — saves a noticeable
  // delay on the first bot move.
  useEffect(() => {
    preload().catch((err) => console.warn('AlphaZero preload failed', err))
  }, [])

  async function handleDrop({ from, to }) {
    const game = gameRef.current
    if (isGameOver(game) || thinking) return false

    if (!tryMove(game, from, to)) return false

    setFen(game.fen())
    if (isGameOver(game)) return true

    setThinking(true)
    try {
      const botMove = await bestMove(game.fen())
      if (botMove) {
        applyMove(game, botMove)
        setFen(game.fen())
      }
    } finally {
      setThinking(false)
    }
    return true
  }

  function newGame() {
    if (thinking) return
    gameRef.current = createGame()
    setFen(gameRef.current.fen())
  }

  const game = gameRef.current
  const interactive = getTurn(game) === HUMAN && !isGameOver(game)

  return (
    <main className="app">
      <h1>Chess</h1>
      <Status message={statusMessage(game, thinking)} />
      <Board fen={fen} interactive={interactive} onDrop={handleDrop} />
      <button className="new-game" onClick={newGame} disabled={thinking}>
        New Game
      </button>
    </main>
  )
}
