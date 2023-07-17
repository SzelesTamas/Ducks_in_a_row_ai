"""In this file I define the agent class and the agent's methods."""
from ducks_in_a_row import Board
import numpy as np
import sys

sys.setrecursionlimit(10000000)

class Agent:
    """This is a class for a Minimax agent for the ducks in a row game."""
    def __init__(self, valueNetwork):
        """Initializes the agent.
        
        Args:
            valueNetwork (torch.nn.Module): The value network to use for the agent.
        """
        self.valueNetwork = valueNetwork
    
    def minimax(self, state, player, maxDepth, _count, _depth=0):
        """This function implements the minimax algorithm for the ducks in a row game.
            Returns the value for a position, 1 is the win for player1, -1 is the win for player2.
            
        Args:
            state (np.array): The current state of the game.
            player (int): The player for whom the best move is to be found.
            maxDepth (int): The maximum depth to search to.
            _count (list): A list containing the number of nodes visited leaf nodes so far.
            _depth (int): The current depth of the search.
            
        Returns:
           float: The value of the position for the players, 1 is win for player 1, -1 is win for player 2.       
        """
        
        _count[0] += 1
        winner = Board.getWinner(state)
        if(winner != 0):
            return 1 if winner == 1 else -1
        
        if(_depth == maxDepth):
            val = self.valueNetwork(np.array([Board.getStateForPlayer(state, player)])).cpu().detach().numpy()[0][0]
            if(player == 2):
                val *= -1
            return val
        
        nextStates = []
        nextPlayer = Board.getNextPlayer(player)
        for moveInd in Board.getValidMoves(state, player):
            nextState = Board.getNextState(state, moveInd)
            nextStates.append(nextState)
        nextWins = np.array([self.minimax(nextState, nextPlayer, maxDepth, _count, _depth+1) for nextState in nextStates])
        
        best = max(nextWins) if player == 1 else min(nextWins)
        
        return best
    
    def alphabeta(self, state, player, maxDepth, _count, _depth=0, _alpha=-np.inf, _beta=np.inf):
        """This function implements the minimax algorithm for the ducks in a row game.
            Returns the value for a position, 1 is the win for player1, -1 is the win for player2.
            
        Args:
            state (np.array): The current state of the game.
            player (int): The player for whom the best move is to be found.
            maxDepth (int): The maximum depth to search to.
            _count (list): A list containing the number of nodes visited leaf nodes so far.
            _depth (int): The current depth of the search.
            _alpha (float): The current alpha value.
            _beta (float): The current beta value.
        """
        _count[0] += 1
        winner = Board.getWinner(state)
        if(winner != 0):
            return 1 if winner == 1 else -1
        
        if(_depth == maxDepth):
            val = self.valueNetwork(np.array([Board.getStateForPlayer(state, player)])).cpu().detach().numpy()[0][0]
            if(player == 2):
                val *= -1
            return val
        
        if(player == 1):
            best = -np.inf
            nextStates = [Board.getNextState(state, moveInd) for moveInd in Board.getValidMoves(state, player)]
            nextStates.sort(key=lambda x: self.valueNetwork(np.array([Board.getStateForPlayer(x, 2)])).cpu().detach().numpy()[0][0])
            for nextState in nextStates:
                best = max(best, self.alphabeta(nextState, Board.getNextPlayer(player), maxDepth, _count, _depth+1, _alpha, _beta))
                _alpha = max(_alpha, best)
                if(_beta <= _alpha):
                    break
        else:
            best = np.inf
            nextStates = [Board.getNextState(state, moveInd) for moveInd in Board.getValidMoves(state, player)]
            nextStates.sort(key=lambda x: self.valueNetwork(np.array([Board.getStateForPlayer(x, 1)])).cpu().detach().numpy()[0][0])
            for nextState in nextStates:
                best = min(best, self.alphabeta(nextState, Board.getNextPlayer(player), maxDepth, _count, _depth+1, _alpha, _beta))
                _beta = min(_beta, best)
                if(_beta <= _alpha):
                    break
                
        return best
        
    def getBestMove(self, state, player, depth=3, debug=False):
        """Returns the best move for the player in the given state.
        
        Args:
            state (np.array): The current state of the game.
            player (int): The player for whom the best move is to be found.
            depth (int): The maximum depth to search to.
            debug (bool): Whether to print debug information.
        
        Returns:
            int: The best move for the player in the given state as an index.
        """
        validMoves = Board.getValidMoves(state, player)
        nextStates = [Board.getNextState(state, moveInd) for moveInd in validMoves]
        nextPlayer = Board.getNextPlayer(player)
        count = [0]
        nextWins = np.array([self.alphabeta(nextState, nextPlayer, depth, count) for nextState in nextStates])
        bestMove = validMoves[np.argmax(nextWins) if player == 1 else np.argmin(nextWins)]
        if(debug):
            if(player == 2):
                nextWins *= -1
            for win, move in zip(nextWins, validMoves):
                print("Move: ", Board.indexToMove(move), " Win probability: ", (win+1)/2)
            print("Count: ", count[0])
            print("Value of position: ", self.valueNetwork(np.array([Board.getStateForPlayer(state, player)])).cpu().detach().numpy()[0][0])
        return bestMove
    
    def getTrainingMove(self, state, player, depth, debug=False):
        """Returns a randomly sampled move for the player in the given state.
        The probability of a move being selected is proportional to the value of the position after the move.
        
        Args:
            state (np.array): The current state of the game.
            player (int): The player for whom the best move is to be found.
            depth (int): The maximum depth to search to.
            debug (bool): Whether to print debug information.
        
        Returns:
            int: The best move for the player in the given state as an index.
        """
        validMoves = Board.getValidMoves(state, player)
        nextStates = [Board.getNextState(state, moveInd) for moveInd in validMoves]
        nextPlayer = Board.getNextPlayer(player)
        count = [0]
        nextWins = np.array([self.alphabeta(nextState, nextPlayer, depth, count) for nextState in nextStates], dtype=np.float32)
        if(player == 2):
            nextWins *= -1
        nextWins += 10.0
        nextWins /= np.sum(nextWins)
        bestMove = np.random.choice(validMoves, size=1, p=nextWins)[0]
        if(debug):
            for win, move in zip(nextWins, validMoves):
                print("Move: ", Board.indexToMove(move), " Win probability: ", (win+1)/2)
            print("Count: ", count[0])
            print("Value of position: ", self.valueNetwork(np.array([Board.getStateForPlayer(state, player)])).cpu().detach().numpy()[0][0])
        return bestMove