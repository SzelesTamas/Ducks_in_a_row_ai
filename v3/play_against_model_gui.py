"""When running this file the user can play against a vanilla Monte Carlo Tree Search a match of Ducks in a row."""
from typing import Any, Literal
from ducks_in_a_row import Board
from alphazero_agent import AlphaZeroAgent
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from time import sleep
from math import sqrt


class Game:
    """This is a class for managing the MCTS Agent and creating a GUI."""

    def __init__(self, screen, agentInd: int = 2):
        """Initializes the Game class.

        Args:
            screen (Any): Pygame display to draw the game on.
            agentInd (int): The index of the MCTS Agent.
        """

        # GUI parameters
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.linewidthPercent = 0.01
        self.outsideSquareSize = min(self.width, self.height) / 5
        self.innerSquareSize = self.outsideSquareSize * (1 - self.linewidthPercent * 2)
        self.boardRect = [
            [None for y in range(5)] for x in range(5)
        ]  # list to store the board square objects in
        self.selectedSquares = []  # clicked squares, the length of it is at most 2
        self.font = pygame.font.SysFont("Comic Sans MS", 40)

        # game parameteres
        self.board = Board()
        self.agentInd = agentInd
        self.agent = AlphaZeroAgent(self.board, agentInd, 1.4, simulationCount=1000)

    def drawBoard(self):
        """Draws the board on the screen."""

        pygame.draw.rect(
            self.screen, (0, 0, 0), (0, 0, self.width, self.height), width=0
        )
        for x in range(5):
            for y in range(5):
                self.boardRect[x][y] = pygame.draw.rect(
                    self.screen,
                    (176, 224, 230),
                    (
                        y * self.outsideSquareSize
                        + self.outsideSquareSize * self.linewidthPercent,
                        x * self.outsideSquareSize
                        + self.outsideSquareSize * self.linewidthPercent,
                        self.innerSquareSize,
                        self.innerSquareSize,
                    ),
                    width=0,
                )

    def drawPieces(self):
        """Draws the pieces on the board as circles."""

        for x in range(5):
            for y in range(5):
                if self.board.state[x][y] == 1:
                    color = (0, 0, 0)
                elif self.board.state[x][y] == 2:
                    color = (255, 255, 255)
                else:
                    color = None

                if color is not None:
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (
                            y * self.outsideSquareSize + self.outsideSquareSize / 2,
                            x * self.outsideSquareSize + self.outsideSquareSize / 2,
                        ),
                        self.innerSquareSize / 2 * 0.8,
                        width=0,
                    )

    def checkGameOver(self):
        """Checks if the game is over and stops the game if it is

        Returns:
            bool: True if the game is not over, False if it is
        """

        ret = Board.getWinner(self.board.state)
        if ret == 1:
            text = self.font.render(
                "Player 1(black) wins!", False, (0, 0, 0), (255, 255, 255)
            )
            self.screen.blit(text, (0, 0))
        elif ret == 2:
            text = self.font.render(
                "Player 2(white) wins!", False, (0, 0, 0), (255, 255, 255)
            )
            self.screen.blit(text, (0, 0))

        return ret == 0

    def getSquare(self, pos):
        """Gets the square at a position returns None if no square is found.

        Args:
            pos (tuple): The position in question.

        Returns:
            tuple: (x, y) coordinate of the clicked square. None if there is no square.
        """
        for x in range(5):
            for y in range(5):
                if self.boardRect[x][y].collidepoint(pos):
                    return (x, y)
        return None

    def checkClick(self, pos):
        """Checks if a square is clicked and appends it to the selected squares list.

        Args:
            pos (tuple): The position of the click.
        """

        # get the clicked square
        square = self.getSquare(pos)
        if square is not None:
            if len(self.selectedSquares) > 0:
                if (
                    self.selectedSquares[-1][0] == square[0]
                    and self.selectedSquares[-1][1] == square[1]
                ):
                    pass
                else:
                    self.selectedSquares.append(square)
            else:
                self.selectedSquares.append(square)

    def checkRelease(self, pos):
        """Checks where the mouse is released and if the click and the release square are the same does nothing. If the clicked and the released square are different and it's a valid move it makes a move.

        Args:
            pos: Position of the release on the screen.
        """
        if self.board.onTurn == self.agentInd:
            return
        if len(self.selectedSquares) == 0:
            return

        square = self.getSquare(pos)
        if square is not None:
            if len(self.selectedSquares) == 1:
                if (
                    square[0] == self.selectedSquares[0][0]
                    and square[1] == self.selectedSquares[0][1]
                ):
                    pass
                else:
                    validMoves = Board.getValidMoves(
                        self.board.state, self.board.onTurn
                    )
                    move = (
                        self.selectedSquares[0][0],
                        self.selectedSquares[0][1],
                        square[0],
                        square[1],
                    )
                    if move in validMoves:
                        self.board.makeMove(move)
                    self.selectedSquares = []
            else:
                if (
                    square[0] == self.selectedSquares[0][0]
                    and square[1] == self.selectedSquares[0][1]
                ):
                    pass
                else:
                    validMoves = Board.getValidMoves(
                        self.board.state, self.board.onTurn
                    )
                    move = (
                        self.selectedSquares[0][0],
                        self.selectedSquares[0][1],
                        self.selectedSquares[1][0],
                        self.selectedSquares[1][1],
                    )
                    if move in validMoves:
                        self.board.makeMove(move)
                    self.selectedSquares = []

    def startGame(self):
        """Starts the game and runs it until termination."""

        # draw the game
        self.drawBoard()
        self.drawPieces()
        pygame.display.update()

        running = True
        mousePressed = False
        while running:
            # check the user/model input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.board.onTurn == self.agentInd:
                    # agents turn
                    print("Agent is thinking...")
                    move = self.agent.getMove(self.board.state)
                    self.board.makeMove(move)
                else:
                    # human turn
                    if event.type == pygame.MOUSEBUTTONDOWN and not mousePressed:
                        mousePressed = True
                        self.checkClick(pygame.mouse.get_pos())
                    if event.type == pygame.MOUSEBUTTONUP:
                        mousePressed = False
                        self.checkRelease(pygame.mouse.get_pos())

            # draw the game
            self.drawBoard()
            self.drawPieces()
            # check if the game is over
            running = running and self.checkGameOver()

            pygame.display.update()
        sleep(3)


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((500, 500))
    game = Game(screen=screen, agentInd=2)
    print("starting new game...")
    game.startGame()
