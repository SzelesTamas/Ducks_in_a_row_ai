"""When running this file the user can play against an Alphazero like agent a match of Ducks in a row."""
from typing import Any, Literal
from ducks_in_a_row import Board
from agent import Node
from neural_networks import ValueNetwork, PolicyNetwork
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from time import sleep
from math import sqrt

import random
import numpy as np
import torch


class Game:
    """This is a class for managing the MCTS Agent and creating a GUI."""

    def __init__(
        self,
        screen,
        agentInd: int = 2,
        simulationCount: int = 1000,
        valueNetworkPath: str = None,
        policyNetworkPath: str = None,
        explorationConstant: float = 1.4,
    ):
        """Initializes the Game class.

        Args:
            screen (Any): Pygame display to draw the game on.
            agentInd (int): The index of the MCTS Agent.
            simulationCount (int, optional): The number of simulations the MCTS Agent will run. Defaults to 1000.
            valueNetworkPath (str, optional): Path to the value network. Defaults to None.
            policyNetworkPath (str, optional): Path to the policy network. Defaults to None.
            explorationConstant (float, optional): Exploration constant for the MCTS agent. Defaults to 1.4.
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

        self.selectedPiece = None
        self.startSquare = None
        self.font = pygame.font.SysFont("Comic Sans MS", 40)

        # game parameteres
        self.currentState = Board.getStartingState()
        self.agentInd = agentInd
        self.humanInd = Board.getNextPlayer(agentInd)
        self.valueNetwork = ValueNetwork(valueNetworkPath)
        self.policyNetwork = PolicyNetwork(policyNetworkPath)
        self.simulationCount = simulationCount
        self.onTurn = 1

        self.pieces = [
            (0, 0) for _ in range(12)
        ]  # list to store the coordinate of each piece
        self.recalculatePiecePositions()
        self.root = Node(
            self.currentState, 1, None, self.valueNetwork, self.policyNetwork, explorationConstant=explorationConstant
        )

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
        for ind, temp in enumerate(self.pieces):
            x, y = temp
            if ind < 6:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)

            if color is not None:
                pygame.draw.circle(
                    self.screen,
                    color,
                    (x, y),
                    self.innerSquareSize / 2 * 0.8,
                    width=0,
                )

    def checkGameOver(self):
        """Checks if the game is over and stops the game if it is

        Returns:
            bool: True if the game is not over, False if it is
        """

        ret = Board.getWinner(self.currentState)
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
        """Moves the pieces according to the input.

        Args:
            pos (tuple): The coordinate of the mouse.
        """

        if self.selectedPiece is None and self.humanInd == self.onTurn:
            square = self.getSquare(pos)
            if square is None:
                return
            # On this square is a movable piece
            if self.currentState[square[0], square[1]] == self.humanInd:
                # get the corresponding piece
                ind = -1
                for i in range(12):
                    temp = self.getSquare(self.pieces[i])
                    if temp[0] == square[0] and temp[1] == square[1]:
                        ind = i
                        break

                # change the position of the piece
                if ind != -1:
                    self.selectedPiece = ind
                    self.startSquare = self.getSquare(self.pieces[ind])
                    self.pieces[ind] = pos
        else:
            self.pieces[self.selectedPiece] = pos

    def recalculatePiecePositions(self):
        """Recalculates the piece positions based on the current state of the board."""

        wInd = 0
        bInd = 6
        for x in range(5):
            for y in range(5):
                ind = -1
                if self.currentState[x][y] == 1:
                    ind = wInd
                    wInd += 1
                elif self.currentState[x][y] == 2:
                    ind = bInd
                    bInd += 1
                else:
                    continue

                self.pieces[ind] = (
                    y * self.outsideSquareSize + self.outsideSquareSize / 2,
                    x * self.outsideSquareSize + self.outsideSquareSize / 2,
                )

    def checkRelease(self, pos):
        """If the user is on turn and makes a valid move executes it. Otherwise does nothing.

        Args:
            pos (tuple): The coordinates of the mouse.
        """
        square = self.getSquare(pos)
        if (
            square is not None
            and self.currentState[square[0], square[1]] == 0
            and self.onTurn == self.humanInd
            and self.selectedPiece is not None
            and self.startSquare is not None
        ):
            move = (self.startSquare[0], self.startSquare[1], square[0], square[1])
            # check if it's a valid move
            if move in [
                Board.indexToMove(m)
                for m in Board.getValidMoves(self.currentState, self.humanInd)
            ]:
                moveInd = Board.moveToIndex(move)
                self.currentState = Board.getNextState(self.currentState, moveInd)
                self.onTurn = Board.getNextPlayer(self.onTurn)

        self.startSquare = None
        self.selectedPiece = None
        self.recalculatePiecePositions()

    def agentTurn(self):
        """Runs the simulations and then makes the best moves according to the agent."""
        for i in range(self.simulationCount):
            self.root.expandTree()
        moveInd, _ = self.root.getMove(self.currentState, self.agentInd)
        print(f"Agent made move {Board.indexToMove(moveInd)}")
        self.currentState = Board.getNextState(self.currentState, moveInd)
        self.onTurn = Board.getNextPlayer(self.onTurn)
        self.recalculatePiecePositions()

    def playGame(self):
        """Starts the game and runs it until termination."""

        # draw the game
        self.drawBoard()
        self.drawPieces()
        pygame.display.update()

        mouseDown = False
        running = True
        while running:
            # check the user/model input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.onTurn == self.agentInd:
                    # agents turn
                    print("Agent is thinking...")
                    self.agentTurn()
                    print("Your turn!")
                else:
                    # human turn
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.checkClick(pygame.mouse.get_pos())
                        mouseDown = True
                    if event.type == pygame.MOUSEBUTTONUP:
                        self.checkRelease(pygame.mouse.get_pos())
                        mouseDown = False

                    if mouseDown:
                        self.checkClick(pygame.mouse.get_pos())

            # draw the game
            self.drawBoard()
            self.drawPieces()
            # check if the game is over
            running = running and self.checkGameOver()

            pygame.display.update()
        sleep(3)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((500, 500))
    modelPath = "models/v11"
    valueNetworkPath = os.path.join(modelPath, "valueNetwork.pt")
    policyNetworkPath = os.path.join(modelPath, "policyNetwork.pt")
    game = Game(
        screen=screen,
        agentInd=2,
        simulationCount=10000,
        valueNetworkPath=valueNetworkPath,
        policyNetworkPath=policyNetworkPath,
        explorationConstant=1.4
    )
    print("Game started!")
    game.playGame()
