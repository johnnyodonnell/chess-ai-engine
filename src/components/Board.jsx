import { Chessboard } from 'react-chessboard'

// Wraps react-chessboard v5. Orientation is fixed to white. `fen` is the
// position to render; `interactive` (computed by the caller from the live game)
// controls whether dragging is allowed.
// onDrop({ from, to }) -> boolean: returns true if the move was accepted
// (react-chessboard keeps the piece on the target square), false to revert.
export default function Board({ fen, interactive, onDrop }) {
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
