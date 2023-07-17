"""This is a module for implementing a Deep Q-Learning agent for the 'Ducks in a row' game."""
from ducks_in_a_row import Board
import numpy as np
from neural_networks import QNetwork
from math import log, sqrt
from collections import deque

class QAgent:
    """This is a class for implementing an agent that uses the Deep Q-Learning algorithm to play the game."""

    def __init__(
        self,
        network,
        epsilon=0.1,
    ):
        """Initializes the agent.

        Args:
            network (torch.nn.Module): The neural network that the agent uses to make decisions.
            epsilon (float, optional): The probability of choosing a random move. Defaults to 0.1.
        """
        self.network = network
        self.epsilon = epsilon

    def getMove(self, state, player):
        """Returns the move that the agent chooses to make.
        
        Args:
            state (np.array): The current state of the game.
            player (int): The player that is making the move.
        """
        
        # making a random move
        if np.random.random() < self.epsilon:
            return Board.getRandomMove(state, player)
        
        # sampling randomly from a probability distribution
        moves = []
        probs = []
        
        # getting the possible moves
        bestValue = -1
        bestMove = None
        for move in Board.getValidMoves(state, player):
            # getting the value of the next state
            t1 = Board.getStateForPlayer(state, player).flatten()
            t2 = np.array(Board.indexToMove(move)).flatten() / 5
            nnInput = np.concatenate((t1, t2), axis=0)
            value = self.network(nnInput).detach().numpy()
            # if the value is better than the current best value, update the best move
            probs.append(value.item())
            moves.append(move)
                
        # normalizing the probabilities
        moves = np.array(moves)
        probs = np.array(probs)
        probs /= probs.sum()
        
        move = np.random.choice(moves, size=1, p=probs).item()
                
        return move
