"""This is a module for implementing the ducks in a row game as an environment for the RL agent."""
import numpy as np
import random
import time

class Board:
    """Board class for the ducks in a row game. The class only contains static methods"""
    winMasks = None
    
    def precomputeWinMask():
        """Precomputes all the winning positions and stores it in a static variable"""
        if(Board.winMasks is not None):
            return
        print("Precomputing win state masks")
        # after this function Board.winMasks is a numpy array which stores a bitmask for each winning position
        
        winMasks = []
        # row positions
        for i in range(5):
            mask = np.zeros((5, 5))
            mask[i, :4] = 1
            winMasks.append(mask)
            mask = np.zeros((5, 5))
            mask[i, 1:] = 1
            winMasks.append(mask)
            
        # column positions
        for i in range(5):
            mask = np.zeros((5, 5))
            mask[:4, i] = 1
            winMasks.append(mask)
            mask = np.zeros((5, 5))
            mask[1:, i] = 1
            winMasks.append(mask)
            
        diagonals = np.array([
            ((0, 0), (1, 1), (2, 2), (3, 3)),
            ((1, 1), (2, 2), (3, 3), (4, 4)),
            ((0, 4), (1, 3), (2, 2), (3, 1)),
            ((1, 3), (2, 2), (3, 1), (4, 0)),
            ((1, 0), (2, 1), (3, 2), (4, 3)),
            ((0, 1), (1, 2), (2, 3), (3, 4)),
            ((0, 3), (1, 2), (2, 1), (3, 0)),
            ((1, 4), (2, 3), (3, 2), (4, 1)),
        ])
        
        for diagonal in diagonals:
            mask = np.zeros((5, 5))
            diagonal = np.transpose(diagonal)
            mask[diagonal[0], diagonal[1]] = 1
            winMasks.append(mask)
        
        Board.winMasks = np.array(winMasks, dtype=bool)  
   
    def getStartingState():
        """Returns the starting state for a standard Ducks in a row game as a numpy array
        
        Returns:
            np.array: Starting state for a standard Ducks in a row game.
        """
        return np.array(
            np.mat(
                "1 0 2 0 2;\
                 2 0 0 0 1;\
                 1 0 0 0 2;\
                 2 0 0 0 1;\
                 1 0 1 0 2"
            )
        )
    
    def getNextPlayer(player: int) -> int:
        """Returns the next player index.

        Args:
            player (int): The current player index.
        """
        return (player % 2) + 1

    def getWinner(state):
        """Returns the index of the winner, 0 if no winner.

        Args:
            state (np.array): The current state of the board.

        Returns:
            int: The index of the winner, 0 if no winner.
        """
        
        for player in [1, 2]:
            playerMask = state == player
            results = (playerMask & Board.winMasks)
            results = results.sum((1, 2)) >= 4
            results = results.sum()
            if(results > 0):
                return player           
            
        return 0

    def getNextState(state, moveInd):
        """Returns the next state after making a move.

        Args:
            state (np.array): The current state of the board.
            moveInd (int): The index of the move to make

        Returns:
            np.array: The next state of the board.
        """
        
        move = Board.indexToMove(moveInd)
        nextState = state.copy()
        # check if move is valid
        if state[move[0], move[1]] == 0:
            print("Invalid move: No piece to move")
            return nextState
        if state[move[2], move[3]] != 0:
            print("Invalid move: Target square already occupied")
            return nextState
        if(abs(move[0]-move[2]) > 1 or abs(move[1]-move[3]) > 1):
            print("Invalid move: starting and ending square is not valid")

        nextState[move[2], move[3]] = state[move[0], move[1]]
        nextState[move[0], move[1]] = 0
        return nextState

    def getValidMoves(state, player):
        """Returns all valid move indices for the given player.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): The index of the player.

        Returns:
            list: All valid move indices for the given player.
        """
        # check if the game is over or not
        winner = Board.getWinner(state)
        if(winner != 0):
            return []
        
        xMoves = [-1, -1, -1, 0, 1, 1, 1, 0]
        yMoves = [-1, 0, 1, 1, 1, 0, -1, -1]
        moves = list(zip(xMoves, yMoves))

        validMoves = []
        for x0 in range(5):
            for y0 in range(5):
                if state[x0, y0] == player:
                    for xMove, yMove in moves:
                        nextX = x0 + xMove
                        nextY = y0 + yMove
                        if nextX < 0 or nextX >= 5 or nextY < 0 or nextY >= 5:
                            continue
                        if state[nextX, nextY] == 0:
                            validMoves.append(Board.moveToIndex((x0, y0, nextX, nextY)))
        return validMoves

    def getRandomMove(state, player):
        """Returns a random move index for the given player.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): The index of the player.

        Returns:
            int: The index of a random valid move.
        """
        validMoves = Board.getValidMoves(state, player)
        if(len(validMoves) == 0):
            print("Error: No valid random move to make")
            return -1
        return random.choice(validMoves)

    def getStateForPlayer(state, player):
        """Returns the state as -1s and 1s for the player, 0s for empty space.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): Index of the player.

        Returns:
            np.array: State as -1s for the opponent and 1s for the player, 0s for the empty space.
        """
        return np.where(state == player, 1, np.where(state == 0, 0, -1))

    def indexToMove(index: int):
        """Returns the move from the index in the format (x0, y0, x1, y1).

        Args:
            index (int): The index of the move.

        Returns:
            tuple: The move in the format (x0, y0, x1, y1).
        """
        # get the starting square
        startInd = index // 8
        startX = startInd // 5
        startY = startInd % 5
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        # 0 1 2
        # 7   3
        # 6 5 4
        endX = startX + moves[index % 8][0]
        endY = startY + moves[index % 8][1]
        return (startX, startY, endX, endY)

    def moveToIndex(move: tuple):
        """Returns the index from the move.

        Args:
            move (tuple): The move in the format (x0, y0, x1, y1).

        Returns:
            int: The index of the move.
        """
        startX, startY, endX, endY = move
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        startInd = startX * 5 + startY
        endInd = moves.index((endX - startX, endY - startY))
        return startInd * 8 + endInd

Board.precomputeWinMask()