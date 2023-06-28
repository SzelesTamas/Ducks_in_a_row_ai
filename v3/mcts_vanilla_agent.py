"""This is a modul for implementing a vanilla MCTS(without neural network) agent for the Ducks in a row game."""
import numpy as np
from ducks_in_a_row import Board
from math import sqrt, log
from random import choice
from time import sleep, time
from multiprocessing import Process, Manager


class Node:
    """This is a class for implementing a node of the MCTS tree.
    Keeps track of the state of the node, the number of wins and the number of visits.
    """

    def __init__(self, state, player=1, parent=None, endNode=False, resultingMove=None):
        """Initializes the node.

        Args:
            state (numpy.array): The state of the environment.
            player (int): Index of the player about to turn.
            parent (Node, optional): Parent node of the node. Defaults to None.
            endNode (bool, optional): True if we cannot go further from this node. Defaults to False.
            resultingMove (tuple, optional): The move which created this state in the form of (x0, y0, x1, y1).
        """
        self.state = state
        self.player = player
        self.children = []
        self.wins = 0
        self.visits = 0
        self.endNode = endNode
        self.parent = parent
        self.resultingMove = resultingMove

    def addChild(self, child):
        """Adds a child to the node.

        Args:
            child (Node): The child node.
        """
        self.children.append(child)
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

        while len(self.children) > 0:
            child = self.children.pop()
            child.deleteAllChildren()

    def copy(self):
        """Returns a copy of the node.

        Returns:
            Node: The copy of the node.
        """
        copy = Node(self.state.copy(), self.player, self.parent, self.endNode)
        copy.wins = self.wins
        copy.visits = self.visits
        copy.resultingMove = self.resultingMove
        return copy


class VanillaMCTS:
    """This is a class for implementing a vanilla MCTS(without neural network) agent for the Ducks in a row game."""

    def __init__(
        self,
        env,
        player,
        maxSimulationCount=12,
        rolloutLength=5,
        explorationParameter=1.4,
    ):
        """Initializes the agent.

        Args:
            env (Board): The environment in which the agent will play.
            player (int): The player that the agent will play as.
            maxSimulationCount (int, optional): The maximum number of MCTS simulations the agent will do. Defaults to 12.
            rolloutLength (int, optional): Length of each MCTS rollout in seconds. Defaults to 5.
            explorationParameter (float, optional): The exploration parameter for the UCT score. Defaults to 1.4.
        """
        self.env = env
        self.player = player
        self.root = None
        self.maxSimulationCount = maxSimulationCount
        self.rolloutLength = rolloutLength
        self.explorationParameter = explorationParameter

    def MCTSSimulation(self, node: Node):
        """Does a full MCTS simulation from a given node.

        Args:
            node (Node): The node to start the simulation from.
        """
        selectedNode = self.selectionPhase(node)
        print("Finished selection phase.")
        expandedNode = self.expansionPhase(selectedNode)
        if expandedNode.endNode:
            print(
                "Expanded node is an end node.-----------------------------------------------------"
            )
        print(expandedNode.state)
        print("Finished expansion phase.")
        results = self.rolloutPhase(expandedNode)
        print(f"Finished rollout phase. Results: {results}")
        self.backpropagationPhase(expandedNode, results)
        print("Finished backpropagation phase.")

    def getUCT(self, node: Node):
        """Returns the UCT score for a given node.

        Args:
            node (Node): The node to get the score for.

        Returns:
            float: The UCT score for the node. Infinity if the node has no parent.
        """
        if node.parent == None or node.visits == 0:
            return float("inf")

        if node.player == self.player:
            return node.wins / node.visits + self.explorationParameter * sqrt(
                log(node.parent.visits) / node.visits
            )
        else:
            return (
                1
                - node.wins / node.visits
                + self.explorationParameter
                * sqrt(log(node.parent.visits) / node.visits)
            )

    def selectionPhase(self, node: Node):
        """Goes down in the game tree until it finds a node that has unexpanded children.

        Args:
            node (Node): The node from which the selection phase will start.

        Returns:
            Node: The node that has unexpanded children. None if there is no such node.
        """
        if node.endNode:
            return None

        # check if the node has any unexpanded children
        hasUnexpandedChild = False
        nextStates = Board.getAllNextStates(node.state, node.player)
        for state in nextStates:
            if not node.hasChildState(state):
                hasUnexpandedChild = True
                break

        if hasUnexpandedChild:
            return node

        # select the child with the highest UCT score
        selectedChild = None
        bestUCT = -10000
        for child in node.children:
            newUCT = self.getUCT(child)
            if newUCT > bestUCT:
                bestUCT = newUCT
                selectedChild = child

        return self.selectionPhase(selectedChild)

    def expansionPhase(self, node: Node):
        """Expands a node and returns the new node.

        Args:
            node (Node): The node to expand.

        Returns:
            Node: The expanded node. None if the node can't be expanded.
        """

        if node.endNode:
            return None

        # randomly choose an unvisited node
        validMoves = Board.getValidMoves(node.state, node.player)
        unvisited = []
        for move in validMoves:
            nextState = Board.getNextState(node.state, move)
            if not node.hasChildState(nextState):
                unvisited.append((nextState, move))

        nextState, move = choice(unvisited)
        child = Node(
            nextState,
            player=Board.getNextPlayer(node.player),
            endNode=(Board.getWinner(nextState) != 0),
            resultingMove=move,
        )
        node.addChild(child)
        return node.children[-1]

    def rolloutPhase(self, node: Node):
        """Does rollouts for the length of the period from the given node. Returns the [no. games, p1 wins, p2 wins] list.

        Args:
            node (Node): The node to do the rollout from.

        Returns:
            list: [number of games, number of wins for player 1, number of wins for player 2]
        """

        # do rollouts for the specified time
        rolloutStartTime = time()
        results = [0, 0, 0]
        board = Board(node.player)
        while time() - rolloutStartTime < self.rolloutLength:
            winner = -1
            board.state = node.state.copy()
            board.onTurn = node.player
            for i in range(50):
                validMoves = Board.getValidMoves(board.state, board.onTurn)
                move = choice(validMoves)
                reward, nextState, done, w = board.makeMove(move)
                if done:
                    winner = w
                    break

            if winner == -1:
                continue
            results[0] += 1
            if winner != 0:
                results[winner] += 1

        return results

    def backpropagationPhase(self, node: Node, results: list):
        """Adds the results(n_games, p1 win, p2 win) to the corresponding nodes in the tree.

        Args:
            node (Node): The node to start the backpropagation from.
            results (list): The results of the rollout phase in the form of [number of games, player 1 wins, player 2 wins].
        """

        # from the starting node go through all the ancestors
        currNode = node
        while currNode is not None:
            currNode.wins += results[currNode.player]
            currNode.visits += results[0]
            currNode = currNode.parent

    def getBestMove(self):
        """Returns the best move for the current state on the board. By best we mean the node with the highes UCT score. The root node will be the current state on the board.
        For the exploration it runs a specified number of MCTS Simulations. This function should be called every turn where MCTS Agent is on turn.

        Returns:
            tuple: (x0, y0, x1, y1) the start and the end point of the move
        """

        # TODO: implement a version of the function where it searches the tree for the current state and uses the existing tree

        # select the root from the already existing tree if possible
        if self.root != None:
            grandChildren = []
            for child in self.root.children:
                grandChildren += child.children

            nextRoot = None
            # search for the current state
            for child in grandChildren:
                if np.array_equal(child.state, self.env.state):
                    nextRoot = child
                    break

            # delete all the unnecessary nodes
            if nextRoot != None:
                print("Using the existing tree.")
                # delete the new root from the parent's children
                nextRoot.parent.children.remove(nextRoot)
                nextRoot.parent = None
                self.root.deleteAllChildren()
                self.root = nextRoot
            else:
                self.root = None

        if self.root == None:
            print("Creating a new tree.")
            self.root = Node(self.env.state, self.player)

        if self.player != self.env.onTurn:
            raise Exception(
                "Agent player index not equal to environment current player index"
            )

        # do the simulations
        for i in range(self.maxSimulationCount):
            print(f"Simulation #{i+1}.")
            self.MCTSSimulation(self.root)

        # choose the best move
        moves = []
        for node in self.root.children:
            moves.append(
                (
                    node.resultingMove,
                    node.visits,
                    ((1 - node.wins / node.visits) if node.visits != 0 else 0),
                    self.getUCT(node),
                )
            )
        # sort the moves by their uct score
        moves.sort(key=lambda x: x[3], reverse=True)
        # print the 10 best moves
        for i in range(min(10, len(moves))):
            print(
                f"{i+1}. {moves[i][0]}: Visits-{moves[i][1]} Winrate-{moves[i][2]} UCT-{moves[i][3]}"
            )

        return moves[0][0]
