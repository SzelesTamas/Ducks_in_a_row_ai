"""In this module I implement everything needed for the self play training."""
from ducks_in_a_row import Board
from agent import QAgent
import numpy as np
from collections import deque, namedtuple
import random
import os
from neural_networks import QNetwork
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
            "Experience", field_names=["stateAction", "targetValue"]
        )

    def addData(self, state, action, targetValue):
        """Adds data to the buffer.

        Args:
            state (np.array): The state of the board.
            action (int): The action taken by the agent.
            targetValue (float): The target value of the state (based on sparse reward).
        """
        t1 = state.flatten()
        t2 = np.array(Board.indexToMove(action)).flatten()
        nnInput = np.concatenate((t1, t2), axis=0)
        self.memory.append(self.experience(nnInput, targetValue))

    def sample(self):
        """Randomly samples a batch from the buffer.

        Returns:
            np.array: states and actions of the batch.
            np.array: The target Q values of the batch.
        """
        batch = random.sample(self.memory, self.batchSize)

        statesActions = np.array([exp.stateAction for exp in batch])
        targetValue = np.array([exp.targetValue for exp in batch])

        return statesActions, targetValue

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
            tuple (int, list): The winner of the game, (state, onTurn, move) for each state in the game.
        """
        state = Board.getStartingState()
        onTurn = 1
        history = []
        for i in range(self.maxGameLength):
            if onTurn == 1:
                temp = agent1.getMove(state, onTurn)
                move = temp
            else:
                temp = agent2.getMove(state, onTurn)
                move = temp

            history.append((state, onTurn, move))
            state = Board.getNextState(state, move)
            onTurn = Board.getNextPlayer(onTurn)
            winner = Board.getWinner(state)

            if winner != 0:
                history.append((state, onTurn, None))
                break

        if self.verbose:
            if winner == 0:
                #print("Game finished with a draw")
                pass
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

        # go through the history backwards and calculate the targets
        d1 = 1
        d2 = 1
        added = 0
        for i in range(len(history) - 2, -1, -1):
            if added >= self.addPerGame:
                break
            state, onTurn, move = history[i]

            if onTurn == winner:
                reward = 0.5 + d1 / 2
                d1 *= self.discountFactor
            else:
                reward = 0.5 - d2 / 2
                d2 *= self.discountFactor

            self.replayBuffer.addData(
                Board.getStateForPlayer(state, onTurn), move, reward
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
        maxGameLength=30,
        trainAfter=5,
        batchSize=32,
        bufferSize=100000,
        n_epochs=10,
        epsilonStart=1,
        epsilonEnd=0.1,
        addPerGame=10
    ):
        """Initializes the SelfPlayTrainer. Both the paths should be to a directory, which should contain a file named "valueNetwork.pt" and a file named "policyNetwork.pt".

        Args:
            savePath (str): The path to save the trained agent to.
            sourcePath (str, optional): The path to load the agent from. Defaults to None.
            maxGameLength (int, optional): The maximum length of a game. Defaults to 30.
            trainAfter (int, optional): The number of games to play before training. Defaults to 5.
            batchSize (int, optional): The batch size to use for training. Defaults to 32.
            bufferSize (int, optional): The size of the replay buffer. Defaults to 100000.
            n_epochs (int, optional): The number of epochs to train for. Defaults to 10.
            epsilonStart (float, optional): The starting value for epsilon. Defaults to 1.
            epsilonEnd (float, optional): The ending value for epsilon. Defaults to 0.1.
            addPerGame (int, optional): The number of moves to add to the replay buffer per game. Defaults to 10.
        """

        self.savePath = savePath
        # Initialize the neural network
        if(sourcePath is not None):
            self.network = QNetwork(modelPath=os.path.join(sourcePath, "network.pt"))
        else:
            self.network = QNetwork(hiddenSizes=[50, 50, 50, 50])

        self.maxGameLength = maxGameLength
        self.trainAfter = trainAfter
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.n_epochs = n_epochs
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.addPerGame = addPerGame

        # Initialize the game manager
        self.gameManager = SelfPlayGameManager(
            self.maxGameLength,
            self.batchSize,
            self.bufferSize,
            verbose=True, 
            addPerGame=self.addPerGame,
        )

    def trainForEpisodes(self, numEpisodes):
        """Trains the agent for a number of episodes. Each episode consists of trainAfter games.

        Args:
            numEpisodes (int): The number of episodes to train for.
        """
        epsilonDecay = (self.epsilonEnd / self.epsilonStart) ** (1/numEpisodes-1)
        print("EpsilonDecay:", epsilonDecay)
        for i in range(numEpisodes):
            print(f"Episode {i+1}/{numEpisodes}")
            print("Playing games...")
            self.gameManager.playGames(
                QAgent(self.network, self.epsilonStart*epsilonDecay**i),
                QAgent(self.network, self.epsilonStart*epsilonDecay**i),
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
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        loss = nn.MSELoss()

        # keep track of the moving average of the losses
        losses = []
        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}/{self.n_epochs}")
            for batchNum in range(
                1, len(self.gameManager.replayBuffer) // self.batchSize + 1
            ):
                (
                    statesActions,
                    targetValues,
                ) = self.gameManager.replayBuffer.sample()
                optimizer.zero_grad()

                preds = self.network(statesActions)
                targetValues = torch.tensor(
                    targetValues, dtype=torch.float32
                ).unsqueeze(1)

                lossValue = loss(preds, targetValues)
                
                lossValue.backward()

                optimizer.step()

                losses.append(lossValue.item())

            losses = losses[-100:]
            print(
                f"Loss: {sum(losses) / len(losses)}"
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
            self.network.save(os.path.join(self.savePath, "network.pt"))
        else:
            print("Loading the saved models")
            # load the models
            savedQNetwork = QNetwork(
                modelPath=os.path.join(self.savePath, "network.pt")
            )

            # play games between the two agents as the newly trained agent is the first player
            games1 = self.gameManager.playGames(
                QAgent(self.network, epsilon=0),
                QAgent(savedQNetwork, epsilon=0),
                testGames,
                True,
            )
            games2 = self.gameManager.playGames(
                QAgent(savedQNetwork, epsilon=0),
                QAgent(self.network, epsilon=0),
                testGames,
                True,
            )

            games = [games1[0] + games2[1], games1[1] + games2[0]]

            print(f"Current agent won {games[0]} games and lost {games[1]} games")
            if games[0] > testGames * 2 * 0.53:
                print("New agent is better")
                # save the models
                self.network.save(os.path.join(self.savePath, "network.pt"))
            else:
                print("New agent is not better")


if __name__ == "__main__":
    trainer = SelfPlayTrainer(
        savePath="models/v2",
        sourcePath="models/v2",
        maxGameLength=20,
        trainAfter=200,
        batchSize=32,
        bufferSize=2000,
        n_epochs=80,
        epsilonStart=0.4,
        epsilonEnd=0.1,
        addPerGame=10,
    )
    trainer.trainForEpisodes(150)
