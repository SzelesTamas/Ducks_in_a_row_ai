"""Tests the agent against a random agent."""
from ducks_in_a_row import Board
from agent import Node, Agent
import numpy as np
from collections import deque, namedtuple
import random
import os
from neural_networks import ValueNetwork, PolicyNetwork


class GameManager:
    """This is a class for playing a number of games between a random agent and an AlphazeroAgent."""

    def __init__(
        self,
        maxGameLength=50,
        verbose=False,
    ):
        """Initializes the GameManager class.

        Args:
            maxGameLength (int, optional): The maximum length of a game. Defaults to 50.
            verbose (bool, optional): Whether to print the moves of the game. Defaults to False.
        """
        self.maxGameLength = maxGameLength
        self.verbose = verbose

    def playGame(self, agent, agentInd):
        """Plays a game between the two agents and returns the winner.

        Args:
            agent (Agent): The agent to play against.
            agentInd (int): The index of the agent in the game.

        Returns:
            tuple (int, list): The winner of the game, (state, onTurn, move, visitCount(200 sized list of the visit count of each child node)) for each state in the game.
        """
        agent.clearTree()
        state = Board.getStartingState()
        onTurn = 1
        gameLength = 0
        for i in range(self.maxGameLength):
            if onTurn == agentInd:
                move, _ = agent.getMove(state, onTurn)
            else:
                move = Board.getRandomMove(state, onTurn)

            gameLength += 1
            state = Board.getNextState(state, move)
            onTurn = Board.getNextPlayer(onTurn)
            winner = Board.getWinner(state)

            if winner != 0:
                break

        if self.verbose:
            if winner == 0:
                print("Game finished with a draw")
            else:
                print(f"Game finished with winner: {winner}, game length: {gameLength}")

        return winner

    def playGames(self, agent, agentInd, numGames):
        """Plays a number of games between a random agent and an AlphazeroAgent and returns the number of wins for each agent.

        Args:
            agent (Agent): The agent to play against.
            agentInd (int): The index of the agent in the game.
            numGames (int): The number of games to play.
            addToBuffer (bool, optional): Whether to add the data from the games to the replay buffer. Defaults to True.

        Returns:
            list: [The number of wins for agent1, The number of wins for agent2]
        """
        wins = [0, 0]
        for i in range(numGames):
            if self.verbose:
                print(f"Playing game {i+1}/{numGames} Wins: {wins[0]}-{wins[1]}")
            while True:
                winner = self.playGame(agent, agentInd)
                if winner != 0:
                    break

            wins[winner - 1] += 1

            if self.verbose:
                print("--------------------")

        return wins

    def testAgents(self, numGames, agent):
        """Tests the agent against a random agent.

        Args:
            numGames (int): The number of games to play.
            agent (Agent): The first agent.

        Returns:
            list: [The number of wins for agent1, The number of wins for agent2]
        """
        # play half the games with agent1 as player 1
        games1 = self.playGames(agent, 1, numGames // 2)
        games2 = self.playGames(agent, 2, numGames // 2)
        games = [0, 0]
        games[0] += games1[0] + games2[1]
        games[1] += games1[1] + games2[0]
        return games


if __name__ == "__main__":
    modelPath = "models/v10"
    valueNetworkPath = os.path.join(modelPath, "valueNetwork.pt")
    policyNetworkPath = os.path.join(modelPath, "policyNetwork.pt")

    agent = Agent(
        simulationCount=300,
        policyNetwork=PolicyNetwork(policyNetworkPath),
        valueNetwork=ValueNetwork(valueNetworkPath),
        explorationConstant=3,
    )
    gameManager = GameManager(maxGameLength=50, verbose=True)
    games = gameManager.testAgents(100, agent)
    print(f"Agent wins: {games[0]}, random agent wins: {games[1]}")
