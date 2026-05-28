import { Chessboard } from 'react-chessboard'
import { HUMAN, getTurn, isGameOver } from '../engine/game.js'

// Wraps react-chessboard v5. Orientation is fixed to white. Dragging is
// allowed only when it's the human's turn and the game isn't over.
// onDrop({ from, to }) -> boolean: returns true if the move was accepted
// (react-chessboard keeps the piece on the target square), false to revert.
export default function Board({ fen, onDrop }) {
  const interactive = getTurn(fen) === HUMAN && !isGameOver(fen)

  const options = {
    position: fen,
    boardOrientation: 'white',
    allowDragging: interactive,
    onPieceDrop: ({ sourceSquare, targetSquare }) => {
      if (!targetSquare) return false
      return onDrop({ from: sourceSquare, to: targetSquare })
    },
  }

  return (
    <div className="board">
      <Chessboard options={options} />
    </div>
  )
}
