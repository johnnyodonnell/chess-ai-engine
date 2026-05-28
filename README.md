# Chess AI Engine

A chess web app (React + Vite) where you play White against a bot. The v1 bot
picks a uniform-random legal move — there's no intelligence yet. The repo is
laid out so smarter engines can drop in later by satisfying the same one-line
contract.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` produces a production build in `dist/`.

## The engine

Every engine lives in `src/engine/` and exposes the same function:

```js
bestMove(fen)  // fen: a chess position string  ->  a move object
```

Because the signature is identical, engines are drop-in interchangeable. The
app picks one with a single import in `src/App.jsx`.

| Engine | Approach | Result |
| --- | --- | --- |
| Random | uniform-random legal move | plays anything; trivially beatable |

### Random — `src/engine/random.js`

Five lines: ask `chess.js` for the legal moves in the current position, pick
one uniformly. No search, no evaluation, no learning.

## Project layout

```
src/
  App.jsx              the game UI and turn logic
  components/          Board, Status
  engine/
    game.js            thin chess.js wrapper — rules, status, FEN helpers
    random.js          engine 1 — uniform-random legal move
styles/app.css         visuals
```
