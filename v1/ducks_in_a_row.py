"""
Modul for emulating the game Ducks in a row
"""
import numpy as np
from typing import Literal
    
class Board():
    def __init__(self, state=None, startingPlayer:Literal[1, 2]=1):
        """Initializes a board

        Args:
            state (optional): Initializes a board object to a state. If state is None then to a default state. Defaults to None.
            
        """
        self.state = None
        if(state is None):
            self.state = np.array(np.mat(
                "1 0 2 0 2;\
                 2 0 0 0 1;\
                 1 0 0 0 2;\
                 2 0 0 0 1;\
                 1 0 1 0 2"
            ))
        else:
            self.state = state.copy()
        self.turn = startingPlayer
            
    def nextPlayer(self, player)->int:
        """Returns the next player

        Args:
            player (int): The current player

        Returns:
            int: The next player
        """
        return (player % 2) + 1
        
    def move(self, x1, y1, x2, y2, inPlace=True):
        """Moves a piece from one position to another if inplace is True, otherwise returns a new board

        Args:
            x1 (int): x coordinate of start
            y1 (int): y coordinate of start
            x2 (int): x coordinate of end
            y2 (int): y coordinate of end
            inplace (bool, optional): If True, moves the piece in place. If False, returns a new board. Defaults to True.
        """
        if(inPlace):
            self.state[x2][y2] = self.state[x1][y1]
            self.state[x1][y1] = 0
            self.turn = self.nextPlayer(self.turn)
        else:
            newBoard = Board(self.state, self.nextPlayer(self.turn))
            newBoard.move(x1, y1, x2, y2, inPlace=True)
            return newBoard

    def getStateForPlayer(state:np.ndarray, player:int)->np.ndarray:
        """Returns a state with 1s for the player and -1s for the opponent
        
        Args:
            state (np.ndarray): The current state of a board
            player (int): The player to get the state for
            
        Returns:
            np.ndarray: The state of the board with 1s for the player and -1s for the opponent
        """
        return np.where(state == player, 1, np.where(state == 0, 0, -1))

    def getValidMoves(state:np.ndarray, player:int)->list:
        """Returns a list of all possible moves in the form of tuples (x1, y1, x2, y2)
        
        Args:
            state (np.ndarray): The current state of a board
            player (int): The player to get the moves for
        
        Returns:
            list: List of all possible moves        
        """
        moveX = [0, 1, 1, 1, 0, -1, -1, -1]
        moveY = [1, 1, 0, -1, -1, -1, 0, 1]
        ret = []
        for x in range(5):
            for y in range(5):
                if(state[x][y] != player):
                    continue
                for dx, dy in zip(moveX, moveY):
                    newX = x + dx
                    newY = y + dy
                    if(newX >= 0 and newY >= 0 and newX < 5 and newY < 5 and state[newX][newY] == 0):
                        ret.append((x, y, newX, newY))
        return ret
    
    def gameOver(self)->int:
        """Returns 0 if the game is not over, 1 if player 1 has won, 2 if player 2 has won
        """
        
        # iterate over all rows
        for x in range(5):
            for player in [1, 2]:
                if(np.all(self.state[x, :4] == player) or np.all(self.state[x, 1:] == player)):
                    return player
            
        # iterate over all columns
        for y in range(5):
            for player in [1, 2]:
                if(np.all(self.state[:4, y] == player) or np.all(self.state[1:, y] == player)):
                    return player
            
        # iterate over all diagonals
        diagonals = [((0, 0), (1, 1), (2, 2), (3, 3)), ((1, 1), (2, 2), (3, 3), (4, 4)), \
                     ((0, 4), (1, 3), (2, 2), (3, 1)), ((1, 3), (2, 2), (3, 1), (4, 0)), \
                     ((1, 0), (2, 1), (3, 2), (4, 3)), ((0, 1), (1, 2), (2, 3), (3, 4)), \
                     ((0, 3), (1, 2), (2, 1), (3, 0)), ((1, 4), (2, 3), (3, 2), (4, 1))]
        for diagonal in diagonals:
            for player in [1, 2]:
                wins = True
                for coordinate in diagonal:
                    if(self.state[coordinate] != player):
                        wins = False
                        break
                if(wins):
                    return player
        
        return 0
      
    def __str__(self):
        """Returns a string representation of the board

        Returns:
            str: String representation of the board
        """
        return str(self.state)
    
    def __copy__(self):
        """Copy method for the board

        Returns:
            Board: A copy of the board
        """
        return Board(self, self.turn)