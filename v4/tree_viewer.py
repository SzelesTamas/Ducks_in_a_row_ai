"""This file is for viewing the MCTS tree."""
from typing import Any, Literal
from ducks_in_a_row import Board
from agent import Node
from neural_networks import ValueNetwork, PolicyNetwork
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from time import sleep
from math import sqrt, ceil

import random
import numpy as np
import torch


class TreeViewer:
    """This class is for viewing the MCTS tree.
    It shows the current node we are at and the children of the node in a smaller window.
    """

    def __init__(
        self,
        valueNetwork: ValueNetwork,
        policyNetwork: PolicyNetwork,
        expansionNum: int = 10,
        explorationConstant: float = 1.4,
    ):
        """Constructor for TreeViewer class

        Args:
            valueNetwork (ValueNetwork): Value network for the MCTS agent
            policyNetwork (PolicyNetwork): Policy network for the MCTS agent
            expansionNum (int, optional): Number of nodes to expand. Defaults to 10.
            explorationConstant (float, optional): Exploration constant for the MCTS agent. Defaults to 1.4.
        """
        self.valueNetwork = valueNetwork
        self.policyNetwork = policyNetwork
        self.expansionNum = expansionNum

        self.font = pygame.font.SysFont("Arial", 20)
        self.screen = pygame.display.set_mode((1330, 600))
        self.root = Node(
            Board.getStartingState(),
            1,
            None,
            self.valueNetwork,
            self.policyNetwork,
            explorationConstant=explorationConstant,
        )
        self.currentNode = self.root

        for _ in range(self.expansionNum):
            self.root.expandTree()

        self.possibleNodes = []

    def drawState(
        self, topLeftX, topLeftY, boardSize, state, backgroundColor=(176, 224, 230)
    ):
        """This function draws a board state on the screen.

        Args:
            topLeftX (int): The x coordinate of the top left corner of the board.
            topLeftY (int): The y coordinate of the top left corner of the board.
            boardSize (int): The size of the board in pixels.
            state (list): The board state to draw.
            backgroundColor (tuple, optional): The background color of the board. Defaults to (176, 224, 230).

        Returns:
            pygame.Rect: The rectangle of the board.
        """
        borderWidth = ceil(boardSize * 0.05)
        innerBoardSize = boardSize - 2 * borderWidth
        lineWidth = ceil(innerBoardSize * 0.008)
        gridSize = innerBoardSize // 5
        duckSize = ceil((gridSize - 2 * lineWidth) * 0.3)

        # Draw the background
        ret = pygame.draw.rect(
            self.screen, backgroundColor, (topLeftX, topLeftY, boardSize, boardSize)
        )
        # Draw the border of the board
        # pygame.draw.rect(self.screen, (0, 0, 0), (topLeftX, topLeftY, boardSize, boardSize), width = borderWidth)

        # Draw the grid
        for i in range(5):
            for j in range(5):
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    (
                        topLeftX + borderWidth + i * gridSize,
                        topLeftY + borderWidth + j * gridSize,
                        gridSize,
                        gridSize,
                    ),
                    width=lineWidth,
                )

        # Draw the ducks
        for i in range(5):
            for j in range(5):
                if state[j][i] == 0:
                    continue
                if state[j][i] == 1:
                    # orange
                    color = (255, 165, 0)
                else:
                    # green
                    color = (0, 128, 0)
                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        topLeftX + borderWidth + i * gridSize + gridSize // 2,
                        topLeftY + borderWidth + j * gridSize + gridSize // 2,
                    ),
                    duckSize,
                )

        return ret

    def onClick(self, pos):
        """This function is called when the user clicks on the screen.
            It checks if the user clicked on one of the children or the parent.
            If so, it sets the current node to the clicked node.

        Args:
            pos (tuple): The position of the mouse click.
        """
        for rect, node in self.possibleNodes:
            if rect.collidepoint(pos):
                self.currentNode = node
                self.possibleNodes = []
                return

    def viewTree(self):
        """Starts the tree viewer."""
        while True:
            # get the events
            for event in pygame.event.get():
                # if the user clicks the close button, exit the program
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                # if the user clicks the mouse, check if they clicked on one of the children or the parent
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.onClick(event.pos)
                # if the user presses the space bar, expand the tree
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    for _ in range(self.expansionNum):
                        self.currentNode.expandTree()

            self.possibleNodes = []
            # draw the background
            self.screen.fill((255, 255, 255))

            # draw the current node on the main side
            self.drawState(10, 10, 490, self.currentNode.state)

            # draw the children of the current node on the other side in 7 by 7 grid (there are only at most 48 children)
            # the children are sorted by the number of visits
            childInd = 0
            childList = []
            for child in self.currentNode.children:
                if child is None:
                    continue
                childList.append(child)

            childList.sort(key=lambda x: x.visitCount, reverse=True)
            for child in childList:
                i = childInd % 7
                j = childInd // 7
                childInd += 1
                self.possibleNodes.append(
                    (
                        self.drawState(20 + 490 + i * 70, 10 + j * 70, 70, child.state),
                        child,
                    )
                )

            # draw the parent of the current node
            if self.currentNode.parent is not None:
                self.possibleNodes.append(
                    (
                        self.drawState(
                            20 + 490 + 7 * 70 + 10,
                            10,
                            120,
                            self.currentNode.parent.state,
                        ),
                        self.currentNode.parent,
                    )
                )

            # display the win probability, the number of visits of the current node and the current player
            text = self.font.render(
                "Win probability: " + str(round(self.currentNode.win, 4)),
                True,
                (0, 0, 0),
            )
            self.screen.blit(text, (10, 510))
            text = self.font.render(
                "Number of visits: " + str(self.currentNode.visitCount), True, (0, 0, 0)
            )
            self.screen.blit(text, (10, 540))
            text = self.font.render(
                "Current player: " + str(self.currentNode.player), True, (0, 0, 0)
            )
            self.screen.blit(text, (10, 570))

            # in another column display the tree size from the current node, the probability of the current node from the parent
            treeSize = self.currentNode.getTreeSize()
            text = self.font.render("Tree size: " + str(treeSize), True, (0, 0, 0))
            self.screen.blit(text, (400, 510))
            if self.currentNode.parent is not None:
                prob = self.currentNode.parent.policy[self.currentNode.resultingMove]
                text = self.font.render(
                    "Probability: " + str(round(prob, 4)), True, (0, 0, 0)
                )
                self.screen.blit(text, (400, 540))
            # display the uct value of the current node
            text = self.font.render(
                "UCT value: " + str(round(self.currentNode.getUCT(), 4)),
                True,
                (0, 0, 0),
            )
            self.screen.blit(text, (400, 570))

            pygame.display.update()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    pygame.init()
    pygame.font.init()
    modelPath = "models/v4"
    valueNetworkPath = os.path.join(modelPath, "valueNetwork.pt")
    policyNetworkPath = os.path.join(modelPath, "policyNetwork.pt")

    treeViewer = TreeViewer(
        ValueNetwork(valueNetworkPath), PolicyNetwork(policyNetworkPath), 1000, 1.4
    )
    treeViewer.viewTree()
