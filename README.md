Chess AI
This project is a custom-built chess engine that combines machine learning evaluation with a traditional search tree.
Overview
Uses Python to generate 12 evaluation metrics (piece activity, king safety, development, etc.).


Trained a neural network on millions of evaluated chess positions from the Lichess open database.


Implements a search tree with depth and time controls, selecting the best move by evaluating leaf positions.


Integrated with the Lichess API to play games automatically:


Accepts incoming challenges.


Adapts its move time based on increment and remaining clock.


Features
Neural-network–based position evaluation.


Adjustable search depth and time limit.


Automatic online play through Lichess.

