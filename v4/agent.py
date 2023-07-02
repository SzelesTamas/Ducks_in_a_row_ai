"""This is a module for implementing a Alpha Zero like agent for the 'Ducks in a row' game."""
from ducks_in_a_row import Board
import numpy as np
from neural_networks import ValueNetwork, PolicyNetwork
from math import log, sqrt
from collections import deque


class Node:
    """This is a class for implementing a node in the MCTS tree."""

    def __init__(
        self,
        state,
        player,
        resultingMove,
        valueNetwork,
        policyNetwork,
        explorationConstant=1.4,
    ):
        """Initializes the Node class.

        Args:
            state (np.array): The state which the node represents.
            player (int): The index of the player on turn.
            resultingMove (int): The move that led to this state.
            valueNetwork (ValueNetwork): The value network to use for rollouts.
            policyNetwork (PolicyNetwork): The policy network to use for rollouts.
            explorationConstant (float, optional): The exploration constant for the MCTS.
        """

        self.state = state.copy()
        self.endNode = Board.getWinner(self.state) != 0
        self.player = player
        self.resultingMove = resultingMove
        self.valueNetwork = valueNetwork
        self.policyNetwork = policyNetwork
        self.explorationConstant = explorationConstant

        self.parent = None
        self.validMoves = Board.getValidMoves(self.state, self.player)
        self.children = [None for i in range(200)]
        self.isValidMove = [False for i in range(200)]
        for move in self.validMoves:
            self.isValidMove[move] = True
        self.visitCount = 0
        self.win, self.policy = self.rollout()
        if(self.endNode):
            self.win = 1
            self.value = 1

    def addChild(self, child):
        """Adds a new child to the children.

        Args:
            child (Node): The child to add.
        """
        if self.children[child.resultingMove] is not None:
            print("Error: A child is already there")
            return
        if not (self.isValidMove[child.resultingMove]):
            print("Error: Not valid move for a child")
            return
        if child.player != Board.getNextPlayer(self.player):
            print("Error: Wrong player index.")
            return

        self.children[child.resultingMove] = child
        child.parent = self

    def notFullyExpanded(self):
        """Returns whether the node is fully expanded or not(has any unvisited children or not).

        Returns:
            bool: True if the node has valid moves which lead to unvisited children, False otherwise.
        """

        for move in self.validMoves:
            if self.children[move] is None:
                return True
        return False

    def getNodeForState(self, state, player):
        """Returns the node which matches the criteria using BFS to get the highest matching node in the tree.

        Args:
            state (np.array): Searched state.
            player (int): Index of the player in the searched state.
        """
        if np.array_equal(self.state, state) and self.player == player:
            return self

        ret = None
        queue = deque()
        queue.append(self)
        while len(queue) > 0:
            curr = queue.popleft()
            if np.array_equal(curr.state, state) and curr.player == player:
                ret = curr
                break

            for move in curr.validMoves:
                if curr.children[move] is not None:
                    queue.append(curr.children[move])

        return ret

    def getUCT(self):
        """Returns the UCT value of the node according to the AlphaZero formula.

        Returns:
            float: The UCT value of the node.
        """

        # encourage immediate wins
        if self.endNode:
            return float("inf")

        par = self.parent
        if par == None:
            par = self

        valueScore = self.win / self.visitCount
        priorScore = (
            self.explorationConstant
            * par.policy[self.resultingMove]
            * sqrt(log(par.visitCount) / self.visitCount)
        )

        return valueScore + priorScore

    def rollout(self):
        """Does a rollout from a leaf node.

        Returns:
            tuple (float, np.array): The value and the policy of the rollout according to the neural networks.
        """
        if self.endNode:
            return (1 if Board.getWinner(self.state) == self.player else -1), np.zeros(
                200
            )

        input = np.array([Board.getStateForPlayer(self.state, self.player)])
        value = self.valueNetwork(input)
        policy = self.policyNetwork(input)

        return value.detach().cpu().numpy()[0].item(), policy.detach().cpu().numpy()[0]

    def select(self):
        """Runs the selection part of the MCTS algorithm.

        Returns:
            Node: The node to expand.
        """
        if self.notFullyExpanded():
            return self

        # choose the next node based on the UCT formula
        bestNode = None
        for move in self.validMoves:
            if self.children[move] is not None:
                if bestNode == None or self.children[move].getUCT() > bestNode.getUCT():
                    bestNode = self.children[move]
        return bestNode.select()

    def expand(self):
        """Runs the expansion part of MCTS algorithm.
            Adds the newly created node to the tree and returns it.

        Returns:
            Node: The newly created node.
        """
        # selecting a random valid move to make
        moves = []
        for move in self.validMoves:
            if self.children[move] is None:
                moves.append(move)
        if len(moves) == 0:
            print("Error: no valid move found although the node is not an end node")
        moveIndex = np.random.choice(moves)

        # creating a new node for that random move
        state = Board.getNextState(self.state, moveIndex)
        player = Board.getNextPlayer(self.player)
        child = Node(
            state=state,
            player=player,
            resultingMove=moveIndex,
            valueNetwork=self.valueNetwork,
            policyNetwork=self.policyNetwork,
            explorationConstant=self.explorationConstant,
        )
        child.visitCount = 1
        self.addChild(child)
        return child

    def calculateWinAndVisit(self):
        """Calculates the win and visit of the current node from the child nodes.
            If there are no child nodes it will do nothing.
        """
        count, s = 0, 0
        for move in self.validMoves:
            if(self.children[move] is not None):
                count += 1
                s += self.children[move].win
                self.visitCount += self.children[move].visitCount
                
        self.win = ((s / count) if count > 0 else self.win)

    def backpropagate(self):
        """Recalculates the win and visit in the ancestors of the node.
        """
        self.calculateWinAndVisit()
        if(self.parent is not None):
            self.parent.backpropagate()
        
    def expandTree(self):
        """Expands the tree by running a full MCTS simulation(selection, expansion, rollout, backpropagation).
            The backpropagation goes all the way to the root node even if the function was not started from there.
        """

        # select the node to expand
        toExpand = self.select()
        if toExpand is None:
            print("No node to expand")
            return
        expanded = toExpand.expand() # on the creation of the node the constructor also does the rollout phase
        expanded.backpropagate()
    
    def getMove(self, state, player):
        """Returns a move index for a state. The move is sampled randomly based on the MCTS tree visit count because it should have a good value.

        Args:
            state (np.array): The state to get the move for.
            player (int): The index of the player on turn.
        
        Returns:
            int: The index of the move.
        """
        node = self.getNodeForState(state, player) # get a corresponding node in the tree
        if(node is None):
            # return a random move if the node is not found in the tree
            return Board.getRandomMove(state, player)
        
        visits = [(0 if node.children[move] is None else node.children[move].visitCount) for move in node.validMoves]
        s = np.sum(visits)
        if(s == 0):
            return Board.getRandomMove(state, player)
        visits /= np.sum(visits)
        ind = int(np.random.choice(np.arange(len(node.validMoves)), size=1, p=visits)[0])
        move = node.validMoves[ind]
        return move
        
    def getBestMove(self, state, player):
        """Returns index of the best move according to the agent for a state. Best means the node with the maximum visitCount.
        
        Args:
            state (np.array): The state to get the move for.
            player (int): The index of the player on turn.
        
        Returns:
            int: The index of the best move.
        """
        node = self.getNodeForState(state, player) # get a corresponding node in the tree
        if(node is None):
            # return a random move if the node is not found in the tree
            return Board.getRandomMove(state, player)
        
        visits = [(0 if node.children[move] is None else node.children[move].visitCount) for move in node.validMoves]
        visits /= np.sum(visits)
        ind = visits.argmax()[0]
        move = node.validMoves[ind]
        return move   






