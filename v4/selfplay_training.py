"""In this module I implement everything needed for the self play training."""
from ducks_in_a_row import Board
from agent import Node, Agent
import numpy as np
from collections import deque, namedtuple
import random
import os
from neural_networks import ValueNetwork, PolicyNetwork
import torch
from torch import nn


class ReplayBuffer:
    """This is a class for storing training data from games.
    I saw this class in this blog about AlphaZero: https://medium.com/@_michelangelo_/alphazero-for-dummies-5bcc713fc9c6
    """

    def __init__(self, maxLen=100000, batchSize=32):
        """Initializes the ReplayBuffer class.

        Args:
            maxLen (int, optional): The maximum length of the buffer. Defaults to 100000.
            batchSize (int, optional): The size of the batch to sample from the buffer. Defaults to 32.
        """
        self.maxLen = maxLen
        self.batchSize = batchSize

        self.memory = deque(maxlen=maxLen)
        self.experience = namedtuple(
            "Experience", field_names=["state", "targetValue", "targetPolicy"]
        )

    def addData(self, state, targetValue, targetPolicy):
        """Adds data to the buffer.

        Args:
            state (np.array): The state of the board.
            targetValue (float): The target value of the state (based on sparse reward).
            targetPolicy (np.array): The target policy of the state (based on visit counts).
        """
        self.memory.append(self.experience(state, targetValue, targetPolicy))

    def sample(self):
        """Randomly samples a batch from the buffer.

        Returns:
            np.array: The states of the batch.
            np.array: The target values of the batch.
            np.array: The target policies of the batch.
        """
        batch = random.sample(self.memory, self.batchSize)

        states = np.array([exp.state for exp in batch])
        targetValues = np.array([exp.targetValue for exp in batch])
        targetPolicies = np.array([exp.targetPolicy for exp in batch])

        return states, targetValues, targetPolicies

    def clear(self):
        """Clears the buffer."""
        self.memory.clear()

    def __len__(self):
        """Returns the length of the buffer.

        Returns:
            int: The length of the buffer.
        """
        return len(self.memory)


class SelfPlayGameManager:
    """This is a class for playing a number of games between two AlphazeroAgents and storing the data from the games."""

    def __init__(
        self,
        maxGameLength=50,
        batchSize=32,
        replayBufferMaxSize=100000,
        discountFactor=0.9,
        addPerGame=10,
        verbose=False,
    ):
        """Initializes the GameManager class.

        Args:
            maxGameLength (int, optional): The maximum length of a game. Defaults to 50.
            batchSize (int, optional): The size of the batch to sample from the replay buffer. Defaults to 32.
            replayBufferMaxSize (int, optional): The maximum size of the replay buffer. Defaults to 100000.
            discountFactor (float, optional): The discount factor for the target value. Defaults to 0.9.
            addPerGame (int, optional): The maximum number of states to add to the replay buffer per game. Defaults to 10.
            verbose (bool, optional): Whether to print the moves of the game. Defaults to False.
        """
        self.maxGameLength = maxGameLength
        self.batchSize = batchSize
        self.replayBufferMaxSize = replayBufferMaxSize
        self.discountFactor = discountFactor
        self.addPerGame = addPerGame
        self.verbose = verbose

        self.replayBuffer = ReplayBuffer(
            maxLen=replayBufferMaxSize, batchSize=batchSize
        )

    def playGame(self, agent1, agent2):
        """Plays a game between the two agents and returns the winner.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.

        Returns:
            tuple (int, list): The winner of the game, (state, onTurn, move, visitCount(200 sized list of the visit count of each child node)) for each state in the game.
        """
        agent1.clearTree()
        agent2.clearTree()
        state = Board.getStartingState()
        onTurn = 1
        history = []
        for i in range(self.maxGameLength):
            if onTurn == 1:
                temp = agent1.getMove(state, onTurn)
                move, visits = temp
            else:
                temp = agent2.getMove(state, onTurn)
                move, visits = temp

            history.append((state, onTurn, move, visits))
            state = Board.getNextState(state, move)
            onTurn = Board.getNextPlayer(onTurn)
            winner = Board.getWinner(state)

            if winner != 0:
                history.append((state, onTurn, None, None))
                break

        if self.verbose:
            if winner == 0:
                print("Game finished with a draw")
            else:
                print(
                    f"Game finished with winner: {winner}, history length: {len(history)}"
                )
        if winner == 0:
            history = []
        return (winner, history)

    def addGameToBuffer(self, history, winner, agent1, agent2):
        """This is a function for calculating the targets and adding the data from a game to the replay buffer.

        Args:
            history (list): The history of the game as a list where each element is (state, onTurn, move, visits).
            winner (int): The winner of the game.
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
        """

        # add the last state to the buffer for both players
        temp = np.random.rand(200)
        temp /= np.sum(temp)
        self.replayBuffer.addData(
            Board.getStateForPlayer(history[-1][0], winner), 1, temp
        )
        self.replayBuffer.addData(
            Board.getStateForPlayer(history[-1][0], Board.getNextPlayer(winner)),
            0,
            temp,
        )

        # go through the history backwards and calculate the targets
        added = 0
        for i in range(len(history) - 2, -1, -1):
            if added >= self.addPerGame:
                break
            state, onTurn, move, visits = history[i]

            if onTurn == winner:
                reward = 0.5 + (self.discountFactor ** (len(history) - i - 1)) / 2
            else:
                reward = 0.5 - (self.discountFactor ** (len(history) - i - 1)) / 2

            self.replayBuffer.addData(
                Board.getStateForPlayer(state, onTurn), reward, visits
            )

            added += 1

    def playGames(self, agent1, agent2, numGames, addToBuffer=True):
        """Plays a number of games between the two agents and returns the number of wins for each agent. If addToBuffer is True, the data from the games is added to the replay buffer.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
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
                winner, history = self.playGame(agent1, agent2)
                if winner != 0:
                    break

            # adding the new game to the buffer
            if addToBuffer:
                self.addGameToBuffer(history, winner, agent1, agent2)
            wins[winner - 1] += 1

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

    def clearBuffer(self):
        """Clears the replay buffer."""
        self.replayBuffer.clear()


class SelfPlayTrainer:
    """This is a class for training an agent using self play."""

    def __init__(
        self,
        savePath,
        sourcePath=None,
        simulationCount=100,
        maxGameLength=30,
        trainAfter=5,
        batchSize=32,
        bufferSize=100000,
        n_epochs=10,
        explorationConstant=1.4,
    ):
        """Initializes the SelfPlayTrainer. Both the paths should be to a directory, which should contain a file named "valueNetwork.pt" and a file named "policyNetwork.pt".

        Args:
            savePath (str): The path to save the trained agent to.
            sourcePath (str, optional): The path to load the agent from. Defaults to None.
            simulationCount (int, optional): The number of simulations to run for each move. Defaults to 100.
            maxGameLength (int, optional): The maximum length of a game. Defaults to 30.
            trainAfter (int, optional): The number of games to play before training. Defaults to 5.
            batchSize (int, optional): The batch size to use for training. Defaults to 32.
            bufferSize (int, optional): The size of the replay buffer. Defaults to 100000.
            n_epochs (int, optional): The number of epochs to train for. Defaults to 10.
            explorationConstant (float, optional): The exploration constant to use for MCTS. Defaults to 1.4.
        """

        self.savePath = savePath
        # Initialize the neural networks
        if sourcePath is None:
            self.valueNetwork = ValueNetwork()
            self.policyNetwork = PolicyNetwork()
        else:
            policyPath = os.path.join(sourcePath, "policyNetwork.pt")
            valuePath = os.path.join(sourcePath, "valueNetwork.pt")
            # check if the files exist
            if not os.path.isfile(policyPath):
                raise Exception("policyNetwork.pt not found in sourcePath")
            if not os.path.isfile(valuePath):
                raise Exception("valueNetwork.pt not found in sourcePath")

            self.policyNetwork = PolicyNetwork(modelPath=policyPath)
            self.valueNetwork = ValueNetwork(modelPath=valuePath)

        self.simulationCount = simulationCount
        self.maxGameLength = maxGameLength
        self.trainAfter = trainAfter
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.n_epochs = n_epochs
        self.explorationConstant = explorationConstant

        # Initialize the game manager
        self.gameManager = SelfPlayGameManager(
            self.maxGameLength,
            self.batchSize,
            self.bufferSize,
            addPerGame=10,
            verbose=True,
        )

    def trainForEpisodes(self, numEpisodes):
        """Trains the agent for a number of episodes. Each episode consists of trainAfter games.

        Args:
            numEpisodes (int): The number of episodes to train for.
        """
        for i in range(numEpisodes):
            print(f"Episode {i+1}/{numEpisodes}")
            print("Playing games...")
            self.gameManager.playGames(
                Agent(
                    self.simulationCount,
                    self.policyNetwork,
                    self.valueNetwork,
                    explorationConstant=self.explorationConstant,
                ),
                Agent(
                    self.simulationCount,
                    self.policyNetwork,
                    self.valueNetwork,
                    explorationConstant=self.explorationConstant,
                ),
                self.trainAfter,
                True,
            )
            print(
                f"Training with replay buffer size {len(self.gameManager.replayBuffer)}..."
            )
            self.train()
            # self.gameManager.clearBuffer()
            print("Saving...")
            self.saveAgent()
            print("--------------------")

    def train(self):
        """Trains the agent using the data in the replay buffer of the game manager."""
        policyOptimizer = torch.optim.Adam(self.policyNetwork.parameters(), lr=0.001)
        valueOptimizer = torch.optim.Adam(self.valueNetwork.parameters(), lr=0.001)
        valueLoss = nn.MSELoss()
        policyLoss = nn.CrossEntropyLoss()

        # keep track of the moving average of the losses
        valueLosses = []
        policyLosses = []
        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}/{self.n_epochs}")
            for batchNum in range(
                1, len(self.gameManager.replayBuffer) // self.batchSize + 1
            ):
                (
                    states,
                    targetValues,
                    targetPolicies,
                ) = self.gameManager.replayBuffer.sample()
                valueOptimizer.zero_grad()
                policyOptimizer.zero_grad()

                valuePreds = self.valueNetwork(states)
                policyPreds = self.policyNetwork(states)
                targetValues = torch.tensor(
                    targetValues, dtype=torch.float32
                ).unsqueeze(1)
                targetPolicies = torch.tensor(targetPolicies, dtype=torch.float32)

                vLoss = valueLoss(valuePreds, targetValues)
                pLoss = policyLoss(policyPreds, targetPolicies)

                vLoss.backward()
                pLoss.backward()

                valueOptimizer.step()
                policyOptimizer.step()

                valueLosses.append(vLoss.item())
                policyLosses.append(pLoss.item())

            valueLosses = valueLosses[-100:]
            policyLosses = policyLosses[-100:]
            print(
                f"Value loss: {sum(valueLosses)/len(valueLosses)} Policy loss: {sum(policyLosses)/len(policyLosses)}"
            )

    def saveAgent(self, testGames=25):
        """Compares the agent with the current best agent and saves it if it is better.

        Args:
            testGames (int, optional): The number of games to play to test the agent. Defaults to 25.
        """

        # check if the directory exists
        if not os.path.exists(self.savePath):
            print("Creating directory")
            os.mkdir(self.savePath)
            # save the models
            self.valueNetwork.save(os.path.join(self.savePath, "valueNetwork.pt"))
            self.policyNetwork.save(os.path.join(self.savePath, "policyNetwork.pt"))
        else:
            print("Loading the saved models")
            # load the models
            savedValueNetwork = ValueNetwork(
                modelPath=os.path.join(self.savePath, "valueNetwork.pt")
            )
            savedPolicyNetwork = PolicyNetwork(
                modelPath=os.path.join(self.savePath, "policyNetwork.pt")
            )

            # play games between the two agents as the newly trained agent is the first player
            games1 = self.gameManager.playGames(
                Agent(self.simulationCount, self.policyNetwork, self.valueNetwork),
                Agent(self.simulationCount, savedPolicyNetwork, savedValueNetwork),
                testGames,
                True,
            )
            games2 = self.gameManager.playGames(
                Agent(self.simulationCount, savedPolicyNetwork, savedValueNetwork),
                Agent(self.simulationCount, self.policyNetwork, self.valueNetwork),
                testGames,
                True,
            )

            games = [games1[0] + games2[1], games1[1] + games2[0]]

            print(f"Current agent won {games[0]} games and lost {games[1]} games")
            if games[0] > testGames * 2 * 0.53:
                print("New agent is better")
                # save the models
                self.valueNetwork.save(os.path.join(self.savePath, "valueNetwork.pt"))
                self.policyNetwork.save(os.path.join(self.savePath, "policyNetwork.pt"))
            else:
                print("New agent is not better")


if __name__ == "__main__":
    trainer = SelfPlayTrainer(
        "models/v4",
        sourcePath=None,
        simulationCount=500,
        maxGameLength=20,
        trainAfter=30,
        batchSize=100,
        bufferSize=5000,
        n_epochs=30,
    )
    trainer.trainForEpisodes(1000)
