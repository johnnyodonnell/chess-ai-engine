# Chess AI Engine

A chess web app (React + Vite) where you play White against an AlphaZero-style
bot. The bot is a small ResNet with policy-and-value heads, trained from random
init by self-play and run in the browser via `onnxruntime-web` (WebGPU when
available, WASM fallback) with PUCT MCTS at inference time.

## Neural network

```mermaid
flowchart TD
    IN["Input<br/>18 × 8 × 8 board planes"] --> STEM["Stem<br/>3×3 conv → 96 · BN · ReLU"]
    STEM --> TOWER

    subgraph TOWER["Residual tower — 4 × ResBlock (96 filters)"]
        direction TB
        RB1["ResBlock 1"] --> RB2["ResBlock 2"] --> RB3["ResBlock 3"] --> RB4["ResBlock 4"]
    end

    TOWER --> PH
    TOWER --> VH

    subgraph PH["Policy head"]
        direction TB
        P1["1×1 conv → 2 · BN · ReLU"] --> P2["flatten (128)"] --> P3["Linear → 4672"]
    end

    subgraph VH["Value head"]
        direction TB
        V1["1×1 conv → 1 · BN · ReLU"] --> V2["flatten (64)"] --> V3["Linear → 256 · ReLU"] --> V4["Linear → 1 · tanh"]
    end

    PH --> POUT["Policy logits<br/>4672 moves"]
    VH --> VOUT["Value<br/>scalar ∈ [-1, 1]"]
```

A single `ResBlock` is two 3×3 convolutions (96 filters, BN, ReLU) with a skip
connection: `ReLU(BN(conv2(ReLU(BN(conv1(x))))) + x)`. The whole net is
~1.3 M parameters. The policy head emits 4672 move logits; the value head emits
a scalar in `[-1, 1]` from the side-to-move's point of view.
