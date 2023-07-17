"""In this module I implement everything needed for the self play training."""
from ducks_in_a_row import Board
from agent import Agent
import numpy as np
from collections import deque, namedtuple
import random
import os
from neural_networks import ValueNetwork
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
            "Experience", field_names=["state", "targetValue"]
        )

    def addData(self, state, targetValue):
        """Adds data to the buffer.

        Args:
            state (np.array): The state of the board.
            targetValue (float): The target value of the state (based on sparse reward).
        """
        self.memory.append(self.experience(state, targetValue))

    def sample(self):
        """Randomly samples a batch from the buffer.

        Returns:
            np.array: The states of the batch.
            np.array: The target values of the batch.
        """
        batch = random.sample(self.memory, self.batchSize)

        states = np.array([exp.state for exp in batch])
        targetValues = np.array([exp.targetValue for exp in batch])

        return states, targetValues

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
    """This is a class for playing a number of games between two Minimax Agents and storing the data from the games."""

    def __init__(
        self,
        maxGameLength=50,
        batchSize=32,
        replayBufferMaxSize=100000,
        discountFactor=0.9,
        addPerGame=10,
        depth=2,
        verbose=False,
    ):
        """Initializes the GameManager class.

        Args:
            maxGameLength (int, optional): The maximum length of a game. Defaults to 50.
            batchSize (int, optional): The size of the batch to sample from the replay buffer. Defaults to 32.
            replayBufferMaxSize (int, optional): The maximum size of the replay buffer. Defaults to 100000.
            discountFactor (float, optional): The discount factor for the target value. Defaults to 0.9.
            addPerGame (int, optional): The maximum number of states to add to the replay buffer per game. Defaults to 10.
            depth (int, optional): The depth of the Minimax Agent. Defaults to 2.
            verbose (bool, optional): Whether to print the moves of the game. Defaults to False.
        """
        self.maxGameLength = maxGameLength
        self.batchSize = batchSize
        self.replayBufferMaxSize = replayBufferMaxSize
        self.discountFactor = discountFactor
        self.addPerGame = addPerGame
        self.depth = depth
        self.verbose = verbose

        self.replayBuffer = ReplayBuffer(
            maxLen=replayBufferMaxSize, batchSize=batchSize
        )

    def playGame(self, agent1:Agent, agent2:Agent):
        """Plays a game between the two agents and returns the winner.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.

        Returns:
            tuple (int, list): The winner of the game, (state, onTurn, move) for each state in the game.
        """
        state = Board.getStartingState()
        onTurn = 1
        history = []
        for i in range(self.maxGameLength):
            if onTurn == 1:
                move = agent1.getTrainingMove(state, onTurn, self.depth)
            else:
                move = agent2.getTrainingMove(state, onTurn, self.depth)

            history.append((state, onTurn, move))
            state = Board.getNextState(state, move)
            onTurn = Board.getNextPlayer(onTurn)
            winner = Board.getWinner(state)

            if winner != 0:
                history.append((state, onTurn, None))
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

    def addGameToBuffer(self, history, winner):
        """This is a function for calculating the targets and adding the data from a game to the replay buffer.

        Args:
            history (list): The history of the game as a list where each element is (state, onTurn, move, visits).
            winner (int): The winner of the game.
        """

        # add the last state to the buffer for both players
        self.replayBuffer.addData(
            Board.getStateForPlayer(history[-1][0], winner), 1
        )
        self.replayBuffer.addData(
            Board.getStateForPlayer(history[-1][0], Board.getNextPlayer(winner)),
            -1,
        )

        # go through the history backwards and calculate the targets
        added = 0
        for i in range(len(history) - 2, -1, -1):
            if added >= self.addPerGame:
                break
            state, onTurn, move = history[i]

            if onTurn == winner:
                reward = (self.discountFactor ** (len(history) - i - 1))
            else:
                reward = -(self.discountFactor ** (len(history) - i - 1))

            self.replayBuffer.addData(
                Board.getStateForPlayer(state, onTurn), reward
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
                self.addGameToBuffer(history, winner)
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
        valueHiddenLayers=[64, 64],
        depth=2,
        maxGameLength=30,
        trainAfter=5,
        saveAfterEpisode=5,
        batchSize=32,
        bufferSize=100000,
        n_epochs=10,
        discountFactor=0.99,
    ):
        """Initializes the SelfPlayTrainer. Both the paths should be to a directory, which should contain a file named "valueNetwork.pt".

        Args:
            savePath (str): The path to save the trained agent to.
            sourcePath (str, optional): The path to load the agent from. Defaults to None.
            valueHiddenLayers (list, optional): The hidden layers to use for the value network if no source is defined. Defaults to [64, 64].
            depth (int, optional): The maximum depth for the Minimax Agent. Defaults to 2.
            simulationCount (int, optional): The number of simulations to run for each move. Defaults to 100.
            maxGameLength (int, optional): The maximum length of a game. Defaults to 30.
            trainAfter (int, optional): The number of games to play before training. Defaults to 5.
            saveAfterEpisode (int, optional): The number of episodes to play before saving the agent. Defaults to 5.
            batchSize (int, optional): The batch size to use for training. Defaults to 32.
            bufferSize (int, optional): The size of the replay buffer. Defaults to 100000.
            n_epochs (int, optional): The number of epochs to train for. Defaults to 10.
            explorationConstant (float, optional): The exploration constant to use for MCTS. Defaults to 1.4.
            discountFactor (float, optional): The discount factor to use for the rewards. Defaults to 0.99.
        """

        self.savePath = savePath
        # Initialize the neural networks
        if sourcePath is None:
            self.valueNetwork = ValueNetwork(hiddenSizes=valueHiddenLayers)
        else:
            valuePath = os.path.join(sourcePath, "valueNetwork.pt")
            # check if the files exist
            if not os.path.isfile(valuePath):
                raise Exception("valueNetwork.pt not found in sourcePath")

            self.valueNetwork = ValueNetwork(modelPath=valuePath)

        self.depth = depth
        self.maxGameLength = maxGameLength
        self.trainAfter = trainAfter
        self.saveAfterEpisode = saveAfterEpisode
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.n_epochs = n_epochs
        self.discountFactor = discountFactor

        # Initialize the game manager
        self.gameManager = SelfPlayGameManager(
            self.maxGameLength,
            self.batchSize,
            self.bufferSize,
            addPerGame=100,
            verbose=True,
            discountFactor=self.discountFactor,
            depth=self.depth,
        )

    def trainForEpisodes(self, numEpisodes):
        """Trains the agent for a number of episodes. Each episode consists of trainAfter games.

        Args:
            numEpisodes (int): The number of episodes to train for.
        """
        for i in range(1, numEpisodes+1):
            print(f"Episode {i}/{numEpisodes}")
            print("Playing games...")
            self.gameManager.playGames(
                Agent(
                    self.valueNetwork,
                ),
                Agent(
                    self.valueNetwork,
                ),
                self.trainAfter,
                True,
            )
            print(
                f"Training with replay buffer size {len(self.gameManager.replayBuffer)}..."
            )
            self.train()
            # self.gameManager.clearBuffer()
            if(i % self.saveAfterEpisode == 0):
                print("Saving...")
                self.saveAgent()
            print("--------------------")

    def train(self):
        """Trains the agent using the data in the replay buffer of the game manager."""
        valueOptimizer = torch.optim.Adam(self.valueNetwork.parameters(), lr=0.001)
        valueLoss = nn.MSELoss().to(self.valueNetwork.device)

        # keep track of the moving average of the losses
        valueLosses = []
        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}/{self.n_epochs}")
            for batchNum in range(
                1, len(self.gameManager.replayBuffer) // self.batchSize + 1
            ):
                (
                    states,
                    targetValues,
                ) = self.gameManager.replayBuffer.sample()
                valueOptimizer.zero_grad()

                valuePreds = self.valueNetwork(states)
                targetValues = torch.tensor(
                    targetValues, dtype=torch.float32
                ).unsqueeze(1).to(self.valueNetwork.device)

                vLoss = valueLoss(valuePreds, targetValues)

                vLoss.backward()

                valueOptimizer.step()

                valueLosses.append(vLoss.item())

            valueLosses = valueLosses[-100:]
            print(
                f"Value loss: {sum(valueLosses)/len(valueLosses)}"
            )

    def saveAgent(self, testGames=20, addToBuffer=True):
        """Compares the agent with the current best agent and saves it if it is better.

        Args:
            testGames (int, optional): The number of games to play to test the agent. Defaults to 25.
            addToBuffer (bool, optional): Whether to add the games to the replay buffer. Defaults to False.
        """

        # check if the directory exists
        if not os.path.exists(self.savePath):
            print("Creating directory")
            os.mkdir(self.savePath)
            # save the models
            self.valueNetwork.save(os.path.join(self.savePath, "valueNetwork.pt"))
        else:
            print("Loading the saved models")
            # load the models
            savedValueNetwork = ValueNetwork(
                modelPath=os.path.join(self.savePath, "valueNetwork.pt")
            )

            # play games between the two agents as the newly trained agent is the first player
            games1 = self.gameManager.playGames(
                Agent(self.valueNetwork),
                Agent(savedValueNetwork),
                testGames,
                addToBuffer,
            )
            games2 = self.gameManager.playGames(
                Agent(savedValueNetwork),
                Agent(self.valueNetwork),
                testGames,
                addToBuffer,
            )

            games = [games1[0] + games2[1], games1[1] + games2[0]]

            print(f"Current agent won {games[0]} games and lost {games[1]} games")
            if games[0] > testGames * 2 * 0.53:
                print("New agent is better")
                # save the models
                self.valueNetwork.save(os.path.join(self.savePath, "valueNetwork.pt"))
            else:
                print("New agent is not better")


if __name__ == "__main__":
    trainer = SelfPlayTrainer(
        savePath="models/v1",
        sourcePath="models/v1",
        valueHiddenLayers=[100, 100, 100, 100, 100, 100, 100],
        maxGameLength=50,
        trainAfter=5,
        saveAfterEpisode=5,
        batchSize=40,
        bufferSize=500,
        n_epochs=20,
        discountFactor=0.7,
        depth=1
    )
    trainer.trainForEpisodes(1000)
