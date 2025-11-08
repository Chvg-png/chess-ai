*Chess AI — Neural Evaluation Meets Search Logic*

This project is a chess engine that merges traditional search algorithms with a
neural-network–based evaluation function. It can run as a *standalone engine*
or as a *live Lichess bot*, letting players face it in real games.

*Overview*

The system focuses on speed, simplicity, and intelligence. The neural network is small, with about 12 nodes—one for each type of piece, such as white pawns or black knights. Each node measures activity, which is based on how close the piece is to the four central squares of the board. Pieces closer to the center are considered more active because they control more space and can move to more places. Each activity value is multiplied by a learned weight, and all results are added together to produce a final evaluation. This simple setup keeps the model fast and interpretable, allowing the engine to search one move deeper in the same time.

*Architecture and Pipeline*

1) *Data collection*: ~250M analyzed Lichess positions (JSON), each with FEN and
   a Stockfish centipawn evaluation (training target).
2) *Feature extraction*: each position becomes 12 numeric heuristic features
3) *Neural evaluation*: the network outputs a single scalar (predicted
   centipawn advantage for the side to move).
4) *Search integration*: the evaluation feeds a minimax search with alpha–beta
   pruning; optional quiescence search extends volatile lines (e.g., captures).

> **Quiescence search (what it means):**  
> Instead of stopping evaluation in the middle of a tactical fight, the engine continues searching noisy positions (like checks or captures) until things “quiet down.”  
> This prevents large tactical swings from being misjudged by cutting search too early.

*Algorithm and Time Management*

- *Move generation*: python-chess.
- *Search*: minimax + alpha–beta pruning; optional quiescence.
- *Timing*: by default, allocate *70% of increment + 3% of remaining time* to
  the move (tunable in code). Suited for live Lichess play, where latency
  matters.
- *Lichess integration*: via the Berserk API.

*Metrics and Evaluation*

Training uses Stockfish centipawn outputs as ground truth.  
The model’s prediction is compared to Stockfish using numerical error metrics
— primarily **MAE (Mean Absolute Error)** — on held-out positions.  
Lower MAE means the NN more closely matches Stockfish’s judgment.

In practice, the engine plays strongly for its simplicity:
- Consistently defeats *600–800* rated players in 3+2 rapid.
- Moves show human-like priorities: activity, coordination, development.

*Implementation and Usage*

- *Language*: Python
- *Libraries*: PyTorch (neural net), python-chess (moves), Berserk (Lichess API)


*Design Philosophy*

This project explores a middle path between machine learning and classical AI:
efficient, interpretable heuristics guiding a fast search. The
network augments the tree with human-like intuition without overwhelming it.

*Limitations and Future Work*

- No transposition table or iterative deepening (yet).

> **Transposition table (TT — what it means):**  
> A cache that stores previously evaluated positions so the engine doesn’t
> recompute them if reached again through a different move order.  
> This saves time and allows the engine to search deeper.

> **Iterative deepening (ID — what it means):**  
> The engine first searches shallow depths, then gradually increases depth.  
> Earlier searches help order moves better for deeper searches, making
> alpha–beta pruning more effective and improving time management.

- Limited positional/endgame nuance compared to large engines.
- Future: TT + ID, improved quiescence, opening book/endgame tables.
*Resources*

- Lichess Database: https://database.lichess.org/
- Lichess API: https://lichess.org/api
- python-chess: https://python-chess.readthedocs.io/en/latest/
- PyTorch: https://pytorch.org/
- Chess Programming Wiki: https://www.chessprogramming.org/Main_Page
