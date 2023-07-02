"""This is a module for implementing the ducks in a row game as an environment for the RL agent."""
import numpy as np
import random


class Board:
    """Board class for the ducks in a row game."""

    def __init__(self, startingPlayer=1):
        """Initializes the board.

        Args:
            startingPlayer (int): The player that starts the game. 1 or 2.
        """
        self.startingPlayer = startingPlayer  # 1 or 2
        self.moves = []  # list of made moves
        self.startNewGame()

    def startNewGame(self):
        """Resets the board to the starting state and changes the starting player."""
        self.moves = []
        self.onTurn = self.startingPlayer  # 1 or 2
        self.state = np.array(
            np.mat(
                "1 0 2 0 2;\
                 2 0 0 0 1;\
                 1 0 0 0 2;\
                 2 0 0 0 1;\
                 1 0 1 0 2"
            )
        )

    def makeMove(self, move):
        """Makes a move on the board(current state) and changes the current player. It also checks if the game is over and who won.

        Args:
            move (tuple): (x0, y0, x1, y1) where (x0, y0) is the starting position and (x1, y1) is the ending position.

        Returns:
            tuple: (reward, next_state, done, winner)
        """

        # check if move is valid
        if self.state[move[0], move[1]] != self.onTurn:
            print("Invalid move! 1", move)
            return 0, self.state, False, None
        if self.state[move[2], move[3]] != 0:
            print("Invalid move! 2", move)
            return 0, self.state, False, None

        # change the state
        self.state[move[0], move[1]] = 0
        self.state[move[2], move[3]] = self.onTurn

        # add the move to the list of moves
        self.moves.append(move)

        # check if the game is over and who won
        winner = Board.getWinner(self.state)
        done = winner != 0
        reward = 1 if done and winner == self.onTurn else (-1 if done else 0)

        # change the player
        self.onTurn = Board.getNextPlayer(self.onTurn)

        return (reward, self.state.copy(), done, winner)

    """Static methods for the Board class."""

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

        # iterate over all rows
        for x in range(5):
            for player in [1, 2]:
                if np.all(state[x, :4] == player) or np.all(state[x, 1:] == player):
                    return player

        # iterate over all columns
        for y in range(5):
            for player in [1, 2]:
                if np.all(state[:4, y] == player) or np.all(state[1:, y] == player):
                    return player

        # iterate over all diagonals
        diagonals = [
            ((0, 0), (1, 1), (2, 2), (3, 3)),
            ((1, 1), (2, 2), (3, 3), (4, 4)),
            ((0, 4), (1, 3), (2, 2), (3, 1)),
            ((1, 3), (2, 2), (3, 1), (4, 0)),
            ((1, 0), (2, 1), (3, 2), (4, 3)),
            ((0, 1), (1, 2), (2, 3), (3, 4)),
            ((0, 3), (1, 2), (2, 1), (3, 0)),
            ((1, 4), (2, 3), (3, 2), (4, 1)),
        ]
        for diagonal in diagonals:
            for player in [1, 2]:
                wins = True
                for coordinate in diagonal:
                    if state[coordinate] != player:
                        wins = False
                        break
                if wins:
                    return player

        return 0

    def getNextState(state, move):
        """Returns the next state after making a move.

        Args:
            state (np.array): The current state of the board.
            move (tuple): (x0, y0, x1, y1) where (x0, y0) is the starting position and (x1, y1) is the ending position.

        Returns:
            np.array: The next state of the board.
        """
        nextState = state.copy()
        # check if move is valid
        if state[move[0], move[1]] == 0:
            return nextState
        if state[move[2], move[3]] != 0:
            return nextState

        nextState[move[2], move[3]] = state[move[0], move[1]]
        nextState[move[0], move[1]] = 0
        return nextState

    def getValidMoves(state, player):
        """Returns all valid moves for the given player.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): The index of the player.

        Returns:
            list: All valid moves for the given player.
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
                            validMoves.append((x0, y0, nextX, nextY))
        return validMoves

    def getRandomMove(state, player):
        """Returns a random move for the given player.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): The index of the player.

        Returns:
            tuple: (x0, y0, x1, y1) where (x0, y0) is the starting position and (x1, y1) is the ending position.
        """
        validMoves = Board.getValidMoves(state, player)
        if(len(validMoves) == 0):
            return []
        return random.choice(validMoves)

    def getAllNextStates(state, player):
        """Returns all the possible next states for the given player.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): The index of the player.
        Returns:
            list: All the possible next states for the given player.
        """
        ret = []
        for move in Board.getValidMoves(state, player):
            ret.append(Board.getNextState(state, move))
        return ret

    def getStateForPlayer(state, player):
        """Returns the state as -1s and 1s for the player, 0s for empty space.

        Args:
            state (np.array): The state we want to evaluate.
            player (int): Index of the player.

        Returns:
            np.array: State as -1s for the opponent and 1s for the player, 0s for the empty space.
        """
        return np.where(state == player, 1, np.where(state == 0, 0, -1))

    def moveFromIndex(index: int):
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

    def indexFromMove(move: tuple):
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
