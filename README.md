   Chess AI — Neural Evaluation Meets Search Logic

This project is a chess engine that merges traditional search algorithms with a neural-network–based evaluation function. It is designed to function both as a standalone engine and as a live bot connected to the Lichess API, allowing players to face it directly in real games.

At its core, the system balances speed, simplicity, and intelligence. The neural network is intentionally lightweight. Containing only 12 parameters that correspond to key chess heuristics such as piece activity, king safety, and development. This design choice reflects a guiding belief behind the engine: that deeper and more efficient search often contributes more to playing strength than an overly complex evaluation model. By minimizing the computational cost of evaluation, the engine gains the freedom to search one depth further within the same time frame, ultimately improving its move accuracy and strategic foresight.

   Architecture and Pipeline

The engine’s structure follows a clear and modular pipeline. It begins with data collection, drawing on a massive dataset of approximately 250 million analyzed Lichess positions. Each position, stored in JSON format, includes a FEN string and a Stockfish evaluation, serving as the training ground for the neural model.

These positions are parsed and preprocessed into a simplified representation that captures 12 heuristic aspects of the position—ranging from piece activity and king safety to central control and mobility. These numeric features form the inputs to the neural network, which produces a single scalar evaluation representing the predicted centipawn advantage for the current player.

Once a position is evaluated, the result feeds into the search algorithm, which explores the game tree to select the best move. The engine employs minimax search with alpha–beta pruning to efficiently explore promising branches while discarding clearly inferior ones. It also includes an optional quiescence search, which extends the exploration of volatile positions (such as those involving captures) to ensure stable evaluations.

   Algorithm and Time Management

When deciding on a move, the engine first generates all legal continuations using the python-chess library. Each resulting position is evaluated by the neural network, and the minimax algorithm propagates the scores up the search tree, alternating between maximizing and minimizing layers depending on the side to move. Alpha–beta pruning significantly reduces the number of nodes evaluated by skipping lines that cannot influence the final decision.

For timing, the engine uses a dynamic allocation strategy that scales with both the clock and the increment. By default, it assigns 70% of the increment plus 3% of the remaining time to each move, though these constants are adjustable within the code. This adaptive approach balances stability with responsiveness, making the engine suitable for live play on Lichess, where network latency can influence move timing.

   Metrics and Evaluation

Training the evaluation network relied on Stockfish’s centipawn outputs as ground truth.

The result is an evaluation function that, while simple, performs remarkably well when integrated with the search system. In testing, the engine has consistently defeated 600–800 rated players on Chess.com in 3+2 rapid games. Its moves exhibit clear understanding of activity, coordination, and development—attributes that emerge naturally from the heuristics it encodes. Though it cannot rival high-level engines, it demonstrates stable, human-like decision-making and a coherent playing style.

  Implementation and Usage

The engine is implemented entirely in Python, with PyTorch handling the neural network, python-chess providing move generation, and Berserk managing communication with the Lichess API.

**Command to download/run**

  Design Philosophy

This project explores a middle ground between machine learning and classical AI. Rather than relying on massive deep networks or exhaustive datasets, it emphasizes efficiency and interpretability. Each of the 12 input heuristics—pawn structure, piece mobility, central control, and others—corresponds to a recognizable chess concept, allowing users to understand how and why the evaluation changes.

In essence, the network augments the search tree with human-like intuition without overwhelming it. This balance of speed and comprehension allows the system to operate in real time while producing strong, explainable moves.

  Limitations and Future Work

The engine’s current simplicity is both its strength and its constraint. Its evaluation lacks positional nuance, its search does not yet use transposition tables or iterative deepening, and its endgame knowledge remains minimal.
 

  Resources

The development of this engine draws on several open resources:

Lichess Database: https://database.lichess.org/
Lichess API: https://lichess.org/api/
Python-Chess Library: https://python-chess.readthedocs.io/en/latest/
PyTorch: https://pytorch.org/
Chess Programming Wiki: https://www.chessprogramming.org/Main_Page


