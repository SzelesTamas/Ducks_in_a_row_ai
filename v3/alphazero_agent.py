"""This is a module for implementing a Alpha Zero like agent for the 'Ducks in a row' game."""
import numpy as np
from ducks_in_a_row import Board
from math import sqrt, log
from random import choice
from time import sleep, time
from multiprocessing import Process, Manager
from neural_networks import ValueNetwork, PolicyNetwork
from queue import PriorityQueue


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
        self.state = state.copy()
        self.player = player
        self.children = [None for i in range(200)]
        self.wins = 0
        self.visits = 0
        self.endNode = endNode
        self.parent = parent
        self.resultingMove = resultingMove

        self.validMoves = [
            Board.indexFromMove(move)
            for move in Board.getValidMoves(self.state, self.player)
        ]

        self.nnValue = 0
        self.nnProbabilities = np.zeros(200)

    def addChild(self, child):
        """Adds a child to the node.

        Args:
            child (Node): The child node.
        """

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

    def detachFromParent(self):
        """Detaches the node from it's parent."""
        if self.parent is not None:
            self.parent.children[self.resultingMove] = None
            self.parent = None

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
            if self.children[i] is None:
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

    def notFullyExpanded(self):
        """Checks if the node is fully expanded (has any unvisited children).

        Returns:
            bool: True if the node is not fully expanded, False otherwise.
        """
        for move in self.validMoves:
            if self.children[move] is None:
                return True
        return False


class AlphaZeroAgent:
    """This is a class for implementing a Alpha Zero like agent for the 'Ducks in a row' game.
    The agent uses Monte Carlo Tree Search and neural networks to play the game.
    It can return the best move for a given state.
    """

    def __init__(
        self,
        player=1,
        explorationConstant=1.4,
        simulationCount=50,
        valueNetworkPath=None,
        policyNetworkPath=None,
        valueNetwork=None,
        policyNetwork=None,
    ):
        """Initializes the agent.

        Args:
            player (int, optional): The index of the player. Defaults to 1.
            explorationConstant (float, optional): The exploration constant used by the agent. Defaults to 1.4.
            simulationCount (int, optional): The number of simulations to do in the MCTS (number of new nodes in the tree). Defaults to 50.
            valueNetworkPath (str, optional): The path to the value neural network. Defaults to None.
            policyNetworkPath (str, optional): The path to the policy neural network. Defaults to None.
            valueNetwork (ValueNetwork, optional): The value neural network. Defaults to None.
            policyNetwork (PolicyNetwork, optional): The policy neural network. Defaults to None.
        """
        # Initialize the parameters
        self.player = player
        self.explorationConstant = explorationConstant
        self.simulationCount = simulationCount
        self.valueNetworkPath = valueNetworkPath
        self.policyNetworkPath = policyNetworkPath

        # Initialize the neural networks
        if valueNetwork is None:
            self.valueNetwork = ValueNetwork(modelPath=valueNetworkPath)
        else:
            self.valueNetwork = valueNetwork
        if policyNetwork is None:
            self.policyNetwork = PolicyNetwork(modelPath=policyNetworkPath)
        else:
            self.policyNetwork = policyNetwork

        # Initialize the root node for the MCTS tree
        self.root = None

    def setRoot(self, state, debug=False):
        """Sets the root node of the tree for the given state.

        Args:
            state (numpy.array): The state of the environment.
        """

        # If the root node is not initialized, initialize it
        if self.root is None:
            self.root = Node(state.copy(), self.player)
            self.root.nnValue, self.root.nnProbabilities = self.rollout(self.root)
            if(debug):
                print("Creating root for the first time")
        else:
            # Search the grandchildren of the root node for the current state
            for move1 in self.root.validMoves:
                if(self.root.children[move1] is None):
                    continue
                for move2 in self.root.children[move1].validMoves:
                    if(self.root.children[move1].children[move2] is None):
                        continue
                    if np.array_equal(self.root.children[move1].children[move2].state, state):
                        self.root = self.root.children[move1].children[move2]
                        if(debug):
                            print("Found root in grandchildren")
                        return
            
            # Search the tree for the current state with a BFS to get the root node
            queue = [self.root]
            newRoot = None
            currSize = 1
            nextSize = 0
            depth = 0
            while len(queue) > 0:
                currSize -= 1
                currNode = queue.pop(0)
                if np.array_equal(currNode.state, state):
                    newRoot = currNode
                    break
                for child in currNode.children:
                    if child is not None:
                        queue.append(child)
                        nextSize += 1
                if currSize == 0:
                    depth += 1
                    currSize = nextSize
                    nextSize = 0
            
            if(newRoot):
                if(debug):
                    print("Found root in tree")
                self.root = newRoot
                return
        
            # Search the grandchildren states of the root node for the current state
            # if found create the node and set it as the root
            for move1Ind in self.root.validMoves:
                childState = Board.getNextState(self.root.state, Board.moveFromIndex(move1Ind))
                validMoves = Board.getValidMoves(childState, Board.getNextPlayer(self.root.player))
                
                for move2 in validMoves:
                    grandChildState = Board.getNextState(childState, move2)
                    if(np.array_equal(grandChildState, state)):
                        # Connect the new node to the tree
                        if(debug):
                            print("Creating root from grandchildren")
                        if(self.root.children[move1Ind] is None):
                            child = Node(childState.copy(), Board.getNextPlayer(self.player), self.root, resultingMove=move1Ind)
                            child.nnValue, child.nnProbabilities = self.rollout(child)
                            self.root.addChild(child)
                        else:
                            child = self.root.children[move1Ind]
                        move2Ind = Board.indexFromMove(move2)
                        self.root = Node(grandChildState.copy(), self.player, child, resultingMove=move2Ind)
                        self.root.nnValue, self.root.nnProbabilities = self.rollout(self.root)
                        child.addChild(self.root)
                        
                        return
            
            if(debug):
                print("Creating root from scratch")
            # If the state is not found in the tree, create a new root node
            self.root = Node(state.copy(), self.player)
            self.root.nnValue, self.root.nnProbabilities = self.rollout(self.root)
            
    def getMove(self, state, debug=False):
        """Returns the best move for the given state according to the agent.

        Args:
            state (numpy.array): The state of the environment.
            debug (bool, optional): If True, the agent will print some info. Defaults to False.

        Returns:
            tuple: (The index of the best move, the policy probabilities for the given state)
        """
        self.setRoot(state, debug=debug)

        # Monte Carlo Tree Search
        # In the previous version I've tried the parallelization of the rollouts but it was slower than the sequential version
        for i in range(1, self.simulationCount + 1):
            self.MCTSSimulation(self.root)  # do a full simulation
            if debug:
                print(
                    f"Finished simulation {i} out of {self.simulationCount}\t\t", end="\r"
                )
                
        if debug:
            print()

        # Get the best move
        # by best we mean the move with the highest number of visits
        # we sample a probability distribution with the number of visits as weights

        probs = self.root.nnProbabilities.copy() / np.sum(self.root.nnProbabilities)
        # the invalid moves should have a probability of 0
        mask = np.zeros(probs.shape)
        for move in self.root.validMoves:
            mask[move] = 1
        probs *= mask  # mask out the invalid moves
        probs /= np.sum(probs)  # renormalize the probabilities

        move = np.random.choice(np.arange(200), p=probs)

        return move, probs

    def getUCT(self, node):
        """Returns the UCT value of the node according to the AlphaZero formula.

        Args:
            node (Node): The node.

        Returns:
            float: The UCT value of the node.
        """
        if node is None:
            return -float("inf")
        if node.visits == 0:
            return float("inf")
        
        # encourage immediate wins
        if node.endNode:
            return float("inf")

        par = node.parent
        if par == None:
            par = node

        valueScore = node.wins / node.visits
        priorScore = (
            self.explorationConstant
            * par.nnProbabilities[node.resultingMove]
            * sqrt(log(par.visits) / node.visits)
        )

        return valueScore + priorScore

    def MCTSSimulation(self, root):
        """Does a MCTS simulation from the given root node.

        Args:
            root (Node): The root node of the tree.
        """
        # select the node to expand
        toExpand = None
        curr = root
        depth = 0
        while curr:
            if curr.endNode:
                print()
                print("Error: End node reached")
                break
            if curr.notFullyExpanded():
                toExpand = curr
                break

            nxt = None
            for move in curr.validMoves:
                if curr.children[move] is None:
                    continue
                if(curr.children[move].endNode):
                    continue
                if nxt is None or self.getUCT(curr.children[move]) > self.getUCT(nxt):
                    nxt = curr.children[move]
            curr = nxt
            depth += 1
            if nxt is None:
                print("Error: no valid move found although the node is not an end node")
                break
            
            
        if toExpand is None:
            print("Error: no node to expand found although the tree is not fully expanded")
            return

        # expand the selected node
        # make a random move and create a new node for it
        # moves which are valid and are not currently expanded are considered
        moves = []
        for move in toExpand.validMoves:
            if(toExpand.children[move] is None):
                moves.append(move)
        if(len(moves) == 0):
            print("Error: no valid move found although the node is not an end node")
        moveIndex = np.random.choice(moves)
        move = Board.moveFromIndex(moveIndex)
        nextState = Board.getNextState(toExpand.state, move)
        player = Board.getNextPlayer(toExpand.player)
        endNode = Board.getWinner(nextState) != 0
        newNode = Node(
            state=nextState, player=player, endNode=endNode, resultingMove=moveIndex
        )
        # do a rollout from the new node
        newNode.nnValue, newNode.nnProbabilities = self.rollout(newNode)
        newNode.wins += newNode.nnValue
        # add the new node to the tree
        toExpand.addChild(newNode)

        # do the backpropagation
        curr = newNode.parent
        while curr:
            curr.visits += 1
            curr.wins += (
                (1 - newNode.wins) if curr.player != newNode.player else newNode.wins
            )
            curr = curr.parent

    def rollout(self, node):
        """Does a rollout from a leaf node.

        Returns:
            tuple (np.array, np.array): The value and the policy of the rollout according to the neural networks.
        """
        if node.endNode:
            return (1 if Board.getWinner(node.state) == node.player else -1), None

        input = np.array([Board.getStateForPlayer(node.state, node.player)])
        value = self.valueNetwork(input)
        policy = self.policyNetwork(input)

        return value.detach().cpu().numpy()[0][0], policy.detach().cpu().numpy()[0]
