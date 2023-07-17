# Ducks in a row ai

In this repository I will try to make an "AI" for playing the game Ducks in a row. 

## Version 1
In this version I tried to play the game with just one policy neural network and a messy implementation.
The agent played random moves.

## Version 2
In this version I tried to implement the Monte Carlo Tree Search algorithm.
The agent sometimes played good moves but many times it blundered.

## Version 3
In this version I implemented an AlphaZero like algorithm.
I think it was a success because after a few episodes of training it learned a little but I want to train it for more time.
In the next version I will try to polish the implementation because in some places it is a bit messy.

## Version 4
This version is just a nicer implementation of the previous version.
I corrected some pretty major bugs and the agent started to converge but I need to run it for some more time.

## Version 5
In this version I'm trying to implement a DQN like algorithm because I think it will be faster to train.
This version is more promising because I hope it learns faster than the previous version.
DQN is a little strong here because it just uses a neural network to approximate the value of a state action pair.
It does not use the Bellman Equation to update the value of a state action pair.
I will try to implement the Bellman Equation in the next version.
This agent did not converge at all.

## Version 6
In this version I implemented a simple MiniMax algorithm with a neural network which gives value for uncertain leaf nodes.
With AlphaBeta pruning it is pretty fast and beats me almost every time. Although this is a solution I'd like to revisit the previous versions and try to make them work.



## Note
I'm reading this book (http://incompleteideas.net/book/RLbook2020.pdf) now and I see how my implementations are off. I will try to fix them.



## Extra Info
As an experiment I am using GitHub Copilot whenever possible
