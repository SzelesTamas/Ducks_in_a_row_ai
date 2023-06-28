"""This is a module for implementing a Alpha Zero like agent for the 'Ducks in a row' game."""
import numpy as np
from ducks_in_a_row import Board
from math import sqrt, log
from random import choice
from time import sleep, time
from multiprocessing import Process, Manager


class Node:
    """This is a class for implementing a node of the MCTS tree.
    Keeps track of the state of the node, the number of wins and the number of visits.
    It also tracks some data about the neural network, like the probabilities of the children and the value of the current state.
    """

    def __init__(self, state, player=1, parent=None, endNode=False, resultingMove=None):
        """Initializes the node.

        Args:
            state (numpy.array): The state of the environment.
            player (int): Index of the player about to turn.
            parent (Node, optional): Parent node of the node. Defaults to None.
            endNode (bool, optional): True if we cannot go further from this node. Defaults to False.
            resultingMove (int, optional): The index of the move that led to this node. Defaults to None.
        """
        self.state = state
        self.player = player
        self.children = [None for i in range(200)]
        self.wins = 0
        self.visits = 0
        self.endNode = endNode
        self.parent = parent
        self.resultingMove = resultingMove
        
        self.nnValue = 0
        self.nnProbabilities = np.zeros(200)

    def addChild(self, child, moveProbability):
        """Adds a child to the node.

        Args:
            child (Node): The child node.
            moveProbability (float): The probability of the move that led to the child.
        """
        self.nnProbabilities[child.resultingMove] = moveProbability
        self.children[child.resultingMove] = child
        child.parent = self
        self.wins += child.wins
        self.visits += child.visits

    def hasChildState(self, state):
        """Checks if the node has a child with the given state.

        Args:
            state (numpy.array): The state of the environment.

        Returns:
            bool: True if the node has a child with the given state, False otherwise.
        """
        for child in self.children:
            if np.array_equal(child.state, state):
                return True
        return False

    def addGames(self, wins, visits):
        """Adds the number of wins and the number of visits to the node.

        Args:
            wins (int): The number of wins.
            visits (int): The number of visits.
        """
        self.wins += wins
        self.visits += visits

    def deleteAllChildren(self):
        """Deletes all the children of the node recursively."""

        for i in range(len(self.children)):
            if(self.children[i] is None):
                continue
            self.children[i].deleteAllChildren()
            del self.children[i]
            self.children[i] = None

    def copy(self):
        """Returns a copy of the node.

        Returns:
            Node: The copy of the node.
        """
        copy = Node(self.state.copy(), self.player, self.parent, self.endNode)
        copy.wins = self.wins
        copy.visits = self.visits
        copy.resultingMove = self.resultingMove
        
        copy.children = self.children.copy()
        copy.nnValue = self.nnValue
        copy.nnProbabilities = self.nnProbabilities.copy()
        return copy

class AlphaZeroAgent:
    pass