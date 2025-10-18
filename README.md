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
   (activity, king safety, central control, mobility, pawn structure, etc.).
3) *Neural evaluation*: the network outputs a single scalar (predicted
   centipawn advantage for the side to move).
4) *Search integration*: the evaluation feeds a minimax search with alpha–beta
   pruning; optional quiescence search extends volatile lines (e.g., captures).

*Algorithm and Time Management*

- *Move generation*: python-chess.
- *Search*: minimax + alpha–beta pruning; optional quiescence.
- *Timing*: by default, allocate *70% of increment + 3% of remaining time* to
  the move (tunable in code). Suited for live Lichess play, where latency
  matters.
- *Lichess integration*: via the Berserk API.

*Metrics and Evaluation*

Training uses Stockfish centipawn outputs as ground truth. In practice, the
engine plays strongly for its simplicity:
- Consistently defeats *600–800* rated players in 3+2 rapid.
- Moves show human-like priorities: activity, coordination, development.

*Implementation and Usage*

- *Language*: Python
- *Libraries*: PyTorch (neural net), python-chess (moves), Berserk (Lichess API)

*Command to download/run*

(Replace the placeholders with your actual commands or scripts.)
- Clone:    git clone <YOUR_REPO_URL>
- Install:  pip install -r requirements.txt
- Local:    python -m your_engine.play
- Lichess:  python -m your_engine.bot   (requires API token/config)

*Design Philosophy*

This project explores a middle path between machine learning and classical AI:
efficient, interpretable heuristics guiding a fast search. Each of the *12*
inputs (e.g., pawn structure, mobility, central control) corresponds to a
recognizable chess idea, so users can reason about changes in evaluation. The
network augments the tree with human-like intuition without overwhelming it.

*Limitations and Future Work*

- No transposition table or iterative deepening (yet).
- Limited positional/endgame nuance compared to large engines.
- Future: TT + ID, improved quiescence, opening book/endgame tables, stronger
  feature set, and lightweight policy/ordering aids.

*Resources*

- Lichess Database: https://database.lichess.org/
- Lichess API: https://lichess.org/api
- python-chess: https://python-chess.readthedocs.io/en/latest/
- PyTorch: https://pytorch.org/
- Chess Programming Wiki: https://www.chessprogramming.org/Main_Page
