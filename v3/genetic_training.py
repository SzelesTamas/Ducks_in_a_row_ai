"""In this file, we will implement the genetic algorithm for training the AlphaZeroAgent.
    It's a long shot but I want to see if it works.
"""
from collections import namedtuple, deque
import numpy as np
from ducks_in_a_row import Board
from alphazero_agent import AlphaZeroAgent
from neural_networks import ValueNetwork, PolicyNetwork
import random
import torch
from torch import nn
import os

class GeneticAgent:
    """This is an extension of the AlphaZeroAgent class for genetic training.
        It has a score attribute which is used for selecting the best agents.
    """
    def __init__(self, score=0, player=1, explorationConstant=1.4, simulationCount=50, valueNetworkPath=None, policyNetworkPath=None, valueNetwork=None, policyNetwork=None):
        """Initializes the GeneticAgent class.
        
        Args:
            score (int, optional): The score of the agent. Defaults to 0.
            player (int, optional): The index of the player. Defaults to 1.
            explorationConstant (float, optional): The exploration constant used by the agent. Defaults to 1.4.
            simulationCount (int, optional): The number of simulations to do in the MCTS (number of new nodes in the tree). Defaults to 50.
            valueNetworkPath (str, optional): The path to the value neural network. Defaults to None.
            policyNetworkPath (str, optional): The path to the policy neural network. Defaults to None.
            valueNetwork (ValueNetwork, optional): The value neural network. Defaults to None.
            policyNetwork (PolicyNetwork, optional): The policy neural network. Defaults to None.
        """
        super().__init__(player, explorationConstant, simulationCount, valueNetworkPath, policyNetworkPath, valueNetwork, policyNetwork)
        
        self.score = score

class GeneticTrainer:
    """This is a class for training a neural net through genetic training (survival of the fittest)."""
    
    def __init__(self, savePath, numAgents=10, numGenerations=10, testGame=50, simulationCount = 100, verbose=False, elitePercentage=0.2, mutationRate=0.1):
        """Initializes the GeneticTrainer class.
        
        Args:
            savePath (str): The path to the folder where the models will be saved.
            numAgents (int, optional): The number of agents in a generation. Defaults to 10.
            numGenerations (int, optional): The number of generations to train for. Defaults to 10.
            testGame (int, optional): The number of games to play to test the agents. Defaults to 50.
            simulationCount (int, optional): The number of simulations to run for each move. Defaults to 100.
            verbose (bool, optional): Whether to print the moves of the game. Defaults to False.
            elitePercentage (float, optional): The percentage of the best agents to keep for the next generation. Defaults to 0.2.
            mutationRate (float, optional): The probability of a mutation. Defaults to 0.1.
        """
        self.savePath = savePath
        self.numAgents = numAgents
        self.numGenerations = numGenerations
        self.testGame = testGame
        self.simulationCount = simulationCount
        self.verbose = verbose
        self.elitePercentage = elitePercentage
        self.mutationRate = mutationRate
        
        self.board = Board()
        self.currentGeneration = [GeneticAgent() for _ in range(self.numAgents)]
        
    def scoreAgents(self):
        """Scores the agents based on the number of wins against a random agent"""
        
        for i in range(self.numAgents):
            gameManager = GameManager(verbose=self.verbose)
            test = gameManager.testAgents(self.testGame, self.currentGeneration[i], AlphaZeroAgent())
            score = test[0] / self.testGame
            self.currentGeneration[i].score = score
        
    def createNewGeneration(self):
        """Creates a new generation from the current generation.
            It assigns a value for every agent based on the number of wins against a random agent.
            Then it creates a new generation by selecting the best agents and mutating them.
        """
        # score the agents
        self.scoreAgents()
        # sort the agents by score (best first)
        self.currentGeneration.sort(key=lambda agent: agent.score, reverse=True)
        
        nextGeneration = []
        # keep the best agents
        eliteCount = int(self.numAgents * self.elitePercentage)
        for i in range(eliteCount):
            nextGeneration.append(self.currentGeneration[i])
            
        # mutate the best agents
        for agent in nextGeneration:
            if random.random() < self.mutationRate:
                agent.valueNetwork = self.mutate(agent.valueNetwork)
                agent.policyNetwork = self.mutate(agent.policyNetwork)
                
    def mutate(self, neuralNetwork):
        """Mutates a neural network by adding a random value to each weight.
        
        Args:
            neuralNetwork: The neural network to mutate.
            
        Returns:
            neuralNetwork: The mutated copy of the neural net
        """
        
        # TODO: implementing
        

        

class GameManager:
    """This is a class for playing a number of games between two AlphazeroAgents and storing the data from the games."""

    def __init__(
        self,
        verbose=False,
    ):
        """Initializes the GameManager class.

        Args:
            verbose (bool, optional): Whether to print the moves of the game. Defaults to False.
        """
        self.verbose = verbose

    def playGame(self, agent1, agent2):
        """Plays a game between the two agents and returns the winner.
        
        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.

        Returns:
            int: The winner of the game.
        """
        self.board.startNewGame()
        agent1.root = None
        agent2.root = None
        while True:
            p = self.board.onTurn
            if self.board.onTurn == 1:
                move, policy = agent1.getMove(self.board.state.copy(), debug=False)
            else:
                move, policy = agent2.getMove(self.board.state.copy(), debug=False)

            _, nextState, done, winner = self.board.makeMove(Board.moveFromIndex(move))

            if done:
                break

        if self.verbose:
            print(
                f"Game finished with winner: {winner}, buffer size: {len(self.replayBuffer)}"
            )
        return winner

    def playGames(self, agent1, agent2, numGames):
        """Plays a number of games between the two agents and returns the number of wins for each agent.
        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
            numGames (int): The number of games to play.

        Returns:
            list: [The number of wins for agent1, The number of wins for agent2]
        """
        wins = [0, 0]
        for i in range(numGames):
            if self.verbose:
                print(f"Playing game {i+1}/{numGames}")
            wins[self.playGame(agent1, agent2) - 1] += 1
            if self.verbose:
                print("--------------------")

        return wins

    def testAgents(self, numGames, agent1, agent2):
        """Tests the agents against each other for a number of games and returns the number of wins for each agent.

        Args:
            numGames (int): The number of games to play.
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.

        Returns:
            list: [The number of wins for agent1, The number of wins for agent2]
        """
        # play half the games with agent1 as player 1
        games1 = self.playGames(agent1, agent2, numGames // 2)
        games2 = self.playGames(agent2, agent1, numGames // 2)
        games = [0, 0]
        games[0] += games1[0] + games2[1]
        games[1] += games1[1] + games2[0]
        return games


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
simulationCount = 1000
trainAfter = 5
savePath = "models/genetic_v1"
