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
        if self.endNode:
            self.win = -1

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
            return self.win / self.visitCount

        par = self.parent
        if par == None:
            return 1

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

        # sample the next node to visit based on it's UCT value
        ucts = []
        maxUct = -1
        nextNode = None
        for move in self.validMoves:
            if self.children[move] is not None:
                temp = self.children[move].getUCT()
                maxUct = max(maxUct, temp)
                ucts.append(temp)
        if len(ucts) == 0:
            return self
        ucts = np.array(ucts)
        maxUct += min(ucts)
        ucts += min(ucts)
        ucts = maxUct - ucts
        ucts /= np.sum(ucts)
        # sample the next node to visit based on it's UCT value
        ind = int(np.random.choice(np.arange(len(ucts)), size=1, p=ucts)[0])
        nextNode = self.children[self.validMoves[ind]]
        return nextNode.select()

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
            winner = Board.getWinner(self.state)
            if winner == 0:
                print("Error: No moves left but no winner")
                print(self.state)
            return None
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

    def backpropagate(self, extraWin, player):
        """Recalculates the win and visit in the ancestors of the node.

        Args:
            extraWin (float): The extra win to add to the node.
            player (int): The player that made the move that lead to the node.
        """
        
        if(self.player == player):
            self.win += extraWin
        else:
            self.win += -extraWin
        
        self.visitCount += 1
        
        if self.parent is not None:
            self.parent.backpropagate(extraWin, player)

    def expandTree(self):
        """Expands the tree by running a full MCTS simulation(selection, expansion, rollout, backpropagation).
        The backpropagation goes all the way to the root node even if the function was not started from there.
        """

        # select the node to expand
        toExpand = self.select()
        if toExpand is None:
            print("No node to expand")
            return
        newChild = toExpand.expand()
        # on the creation of the node the constructor also does the rollout phase
        
        # we are at an end node
        if newChild is None:
            toExpand.backpropagate(1 if Board.getWinner(toExpand.state) == toExpand.player else -1, toExpand.player)
            return
        if(newChild.parent is not None):
            newChild.parent.backpropagate(newChild.win, newChild.player)

    def getMove(self, state, player):
        """Returns a move index for a state. The move is sampled randomly based on the MCTS tree visit count because it should have a good value.

        Args:
            state (np.array): The state to get the move for.
            player (int): The index of the player on turn.

        Returns:
            tuple (int, np.array): The move index and the visit count of each child node.
        """
        node = self.getNodeForState(
            state, player
        )  # get a corresponding node in the tree
        if node is None:
            # return a random move if the node is not found in the tree
            visits = np.random.rand(200)
            visits /= np.sum(visits)
            return Board.getRandomMove(state, player), visits

        visits = np.array(
            [
                (0 if node.children[move] is None else node.children[move].visitCount)
                for move in range(200)
            ],
            dtype=np.float32,
        )
        s = np.sum(visits)
        if s == 0:
            visits = np.random.rand(200)
            visits /= np.sum(visits)
            return Board.getRandomMove(state, player), visits
        visits /= np.sum(visits)
        move = int(np.random.choice(np.arange(200), size=1, p=visits)[0])
        return move, visits

    def getBestMove(self, state, player):
        """Returns index of the best move according to the agent for a state. Best means the node with the maximum visitCount.

        Args:
            state (np.array): The state to get the move for.
            player (int): The index of the player on turn.

        Returns:
            tuple (int, np.array): The index of the move and the visit counts of each child node.
        """
        node = self.getNodeForState(
            state, player
        )  # get a corresponding node in the tree
        if node is None:
            # return a random move if the node is not found in the tree
            visits = np.random.rand(200)
            visits /= np.sum(visits)
            return Board.getRandomMove(state, player), visits

        visits = np.array(
            [
                (0 if node.children[move] is None else node.children[move].visitCount)
                for move in range(200)
            ],
            dtype=np.float32,
        )
        visits /= np.sum(visits)
        move = visits.argmax()[0]
        return move, visits

    def getTreeSize(self):
        """Returns the number of nodes in the tree starting from the current node.

        Returns:
            int: The number of nodes in the tree, including the current node.
        """
        ret = 1
        for move in range(200):
            if self.children[move] is not None:
                ret += self.children[move].getTreeSize()
        return ret


class Agent:
    """This is a class for implementing an agent that uses the MCTS algorithm to play the game."""

    def __init__(
        self,
        simulationCount=1000,
        policyNetwork=None,
        valueNetwork=None,
        explorationConstant=1.4,
    ):
        """Initializes the agent.

        Args:
            simulationCount (int, optional): The number of simulations to run for each move. Defaults to 1000.
            policyNetwork (torch.nn.Module, optional): The policy network. Defaults to None.
            valueNetwork (torch.nn.Module, optional): The value network. Defaults to None.
            explorationConstant (float, optional): The exploration constant for the MCTS algorithm. Defaults to 1.4.
        """
        self.simulationCount = simulationCount
        self.policyNetwork = policyNetwork
        self.valueNetwork = valueNetwork
        self.explorationConstant = explorationConstant

        self.rootNode = Node(
            state=Board.getStartingState(),
            player=1,
            valueNetwork=self.valueNetwork,
            policyNetwork=self.policyNetwork,
            explorationConstant=self.explorationConstant,
            resultingMove=None,
        )

    def clearTree(self):
        """Clears the tree of the agent."""
        self.rootNode = Node(
            state=Board.getStartingState(),
            player=1,
            valueNetwork=self.valueNetwork,
            policyNetwork=self.policyNetwork,
            explorationConstant=1.4,
            resultingMove=None,
        )

    def getMove(self, state, player):
        """Returns a move index for a state after the simulations. The move is sampled randomly based on the MCTS tree visit count because it should have a good value.

        Args:
            state (np.array): The state to get the move for.
            player (int): The index of the player on turn.

        Returns:
            int: The index of the move.
        """
        startFrom = self.rootNode.getNodeForState(state, player)
        if startFrom == None:
            self.rootNode = Node(
                state=state,
                player=player,
                valueNetwork=self.valueNetwork,
                policyNetwork=self.policyNetwork,
                explorationConstant=1.4,
                resultingMove=None,
            )
            startFrom = self.rootNode

        for i in range(self.simulationCount):
            startFrom.expandTree()

        return startFrom.getMove(state, player)
