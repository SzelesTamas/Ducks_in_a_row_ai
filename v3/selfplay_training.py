"""When running this file it will train the AlphaZeroAgent against itself."""
from collections import namedtuple, deque
import numpy as np
from ducks_in_a_row import Board
from alphazero_agent import AlphaZeroAgent
from neural_networks import ValueNetwork, PolicyNetwork
import random
import torch
from torch import nn
import os


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


class GameManager:
    """This is a class for playing a number of games between two AlphazeroAgents and storing the data from the games."""

    def __init__(
        self,
        agent1,
        agent2,
        board,
        replayBuffer,
        winReward=1.0,
        loseReward=-1.0,
        discountFactor=0.9,
        maxMoves=50,
        verbose=False,
    ):
        """Initializes the GameManager class.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
            board (Board): The board object.
            replayBuffer (ReplayBuffer): The replay buffer object.
            winReward (float, optional): The reward for winning a game. Defaults to 1.0.
            loseReward (float, optional): The reward for losing a game. Defaults to -1.0.
            discountFactor (float, optional): The discount factor for the rewards. Defaults to 0.9.
            maxMoves (int, optional): The maximum number of moves in a game. Defaults to 50.
            verbose (bool, optional): Whether to print the moves of the game. Defaults to False.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.replayBuffer = replayBuffer
        self.winReward = winReward
        self.loseReward = loseReward
        self.discountFactor = discountFactor
        self.maxMoves = maxMoves
        self.verbose = verbose

        self.board = board

    def playGame(self):
        """Plays a game between the two agents, calculates the rewards and stores the data in the replay buffer. And returns the winner of the game.
        Because we are dealing with sparse rewards we need to calculate the rewards after the game is finished.
        We use a discount factor for providing extra reward for moves that lead to a win.
        Be careful to use a simulationCount to avoid creating new roots and losing the data from previous moves

        Returns:
            int: The winner of the game.
        """
        self.board.startNewGame()
        self.agent1.root = None
        self.agent2.root = None
        gameStates = [(self.board.state.copy(), self.board.onTurn, None)]
        winner = -1
        for i in range(self.maxMoves):
            p = self.board.onTurn
            if self.board.onTurn == 1:
                move, policy = self.agent1.getMove(self.board.state.copy(), debug=False)
            else:
                move, policy = self.agent2.getMove(self.board.state.copy(), debug=False)

            _, nextState, done, temp = self.board.makeMove(Board.moveFromIndex(move))
            gameStates[-1] = (gameStates[-1][0], gameStates[-1][1], move)
            gameStates.append((nextState.copy(), p, None))

            if done:
                winner = temp
                break
            
        if(winner == -1):
            if(self.verbose):
                print("Draw")
            return -1

        if winner == 1:
            reward1 = self.winReward
            reward2 = self.loseReward
        else:
            reward1 = self.loseReward
            reward2 = self.winReward

        root1 = self.agent1.root
        while root1.parent != None:
            root1 = root1.parent
        root2 = self.agent2.root
        while root2.parent != None:
            root2 = root2.parent

        for ind, val in enumerate(gameStates):
            state, player, move = val
            targetValue = (reward1 if player == 1 else reward2) * (
                self.discountFactor ** (len(gameStates) - ind - 1)
            )
            targetPolicy = np.array(
                [
                    (0 if (child is None) else child.visits)
                    for child in (root1.children if player == 1 else root2.children)
                ]
            )
            if sum(targetPolicy) != 0:
                targetPolicy = targetPolicy / np.sum(targetPolicy)
                stateForPlayer = Board.getStateForPlayer(state, player)
                self.replayBuffer.addData(stateForPlayer, targetValue, targetPolicy)

            if move == None:
                if self.verbose:
                    print("move is None (probably the game is over)")
                break
            root1 = root1.children[move]
            if ind != 0:
                root2 = root2.children[move]

            if root1 == None:
                print("Error: root1 is None ind:", ind)
                break
            if root2 == None:
                print("Error: root2 is None ind:", ind)
                break

        if self.verbose:
            print(
                f"Game finished with winner: {winner}, buffer size: {len(self.replayBuffer)}"
            )
        return winner

    def playGames(self, numGames):
        """Plays a number of games between the two agents and stores the data from the games in the replay buffer. Returns the number of wins for each player.

        Args:
            numGames (int): The number of games to play.

        Returns:
            list: [The number of wins for agent1, The number of wins for agent2]
        """
        wins = [0, 0]
        for i in range(numGames):
            if self.verbose:
                print(f"Playing game {i+1}/{numGames}")
            winner = self.playGame()
            while(winner == -1):
                winner = self.playGame()
            wins[winner-1] += 1
            if self.verbose:
                print("--------------------")

        return wins


class SelfPlayTrainer:
    """This is a class for training a neural net through self play."""

    def __init__(
        self,
        savePath,
        sourcePath=None,
        simulationCount=500,
        trainAfter=5,
        batchSize=32,
        bufferSize=10000,
        n_epochs=10,
    ):
        """Initializes the SelfPlayTrainer class. Both the paths should be to a directory, which should contain a file named "valueNetwork.pt" and a file named "policyNetwork.pt".

        Args:
            savePath (str): The path to save the model to.
            sourcePath (str, optional): The path to load the model from. Defaults to None.
            simulationCount (int, optional): The number of simulations to run for each move. Defaults to 500.
            trainAfter (int, optional): The number of games to play before training the model and cleaning the replayBuffer. Defaults to 5.
            batchSize (int, optional): The batch size for training the model. Defaults to 32.
            bufferSize (int, optional): The size of the replay buffer. Defaults to 10000.
            n_epochs (int, optional): The number of epochs to train the model for. Defaults to 10.
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
        self.trainAfter = trainAfter
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.n_epochs = n_epochs

        # Initialize the replay buffer
        self.replayBuffer = ReplayBuffer(self.bufferSize)

        # Initialize the board
        self.board = Board()

        # Initialize the agents
        self.agent1 = AlphaZeroAgent(
            player=1,
            simulationCount=self.simulationCount,
            valueNetwork=self.valueNetwork,
            policyNetwork=self.policyNetwork,
        )
        self.agent2 = AlphaZeroAgent(
            player=2,
            simulationCount=self.simulationCount,
            valueNetwork=self.valueNetwork,
            policyNetwork=self.policyNetwork,
        )

        # Initialize the game manager
        self.manager = GameManager(
            self.agent1, self.agent2, self.board, self.replayBuffer, verbose=True
        )

        # Initialize the optimizers
        self.policyOptimizer = torch.optim.Adam(
            self.policyNetwork.parameters(), lr=0.001
        )
        self.valueOptimizer = torch.optim.Adam(self.valueNetwork.parameters(), lr=0.001)

        # Initialize the loss functions
        self.valueLoss = nn.MSELoss()
        self.policyLoss = nn.CrossEntropyLoss()

    def trainForEpisodes(self, numEpisodes):
        """Trains the model for a number of episodes and saves the best model after each episode.

        Args:
            numEpisodes (int): The number of episodes to train for.
        """
        for episode in range(1, numEpisodes + 1):
            print(f"Episode {episode}/{numEpisodes}")
            self.manager.playGames(self.trainAfter)
            print("Training the model...")
            self.train()
            print("Testing and saving the model...")
            self.saveBestModel()
            print("Clearing the replay buffer...")
            self.replayBuffer.clear()

    def train(self):
        """Trains the model for a number of epochs (1 epoch means on 1 batch) with the data in the replay buffer."""

        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}/{self.n_epochs}")
            states, targetValues, targetPolicies = self.replayBuffer.sample()
            self.valueOptimizer.zero_grad()
            self.policyOptimizer.zero_grad()
            
            valuePreds = self.valueNetwork(states)
            policyPreds = self.policyNetwork(states)
            targetValues = torch.tensor(targetValues, dtype=torch.float32).unsqueeze(1)
            targetPolicies = torch.tensor(targetPolicies, dtype=torch.float32)

            vLoss = self.valueLoss(valuePreds, targetValues)
            pLoss = self.policyLoss(policyPreds, targetPolicies)

            vLoss.backward()
            pLoss.backward()

            self.valueOptimizer.step()
            self.policyOptimizer.step()
            print(f"Value loss: {vLoss.item()}, Policy loss: {pLoss.item()}")

    def saveBestModel(self):
        """Plays 100 games between the current agent and the saved and saves the model if the agent wins more than 55 games."""

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
            #  games as player 1
            numGames = 25
            agent1 = AlphaZeroAgent(
                player=1,
                simulationCount=self.simulationCount,
                valueNetwork=self.valueNetwork,
                policyNetwork=self.policyNetwork,
            )
            agent2 = AlphaZeroAgent(
                player=2,
                simulationCount=self.simulationCount,
                valueNetwork=savedValueNetwork,
                policyNetwork=savedPolicyNetwork,
            )
            board = Board()
            replayBuffer = ReplayBuffer(1)
            manager = GameManager(agent1, agent2, board, replayBuffer, verbose=True)
            print("Playing games----------------------------")
            games1 = manager.playGames(numGames)
            # games as player 2
            agent1 = AlphaZeroAgent(
                player=1,
                simulationCount=self.simulationCount,
                valueNetwork=savedValueNetwork,
                policyNetwork=savedPolicyNetwork,
            )
            agent2 = AlphaZeroAgent(
                player=2,
                simulationCount=self.simulationCount,
                valueNetwork=self.valueNetwork,
                policyNetwork=self.policyNetwork,
            )
            board = Board()
            replayBuffer = ReplayBuffer(1)
            manager = GameManager(agent1, agent2, board, replayBuffer, verbose=True)
            games2 = manager.playGames(numGames)

            games = [0, 0]  # [currentAgent, savedAgent]
            games[0] += games1[0] + games2[1]
            games[1] += games1[1] + games2[0]
            
            # playing 10 games against a random player
            agent1 = AlphaZeroAgent(
                player=1,
                simulationCount=self.simulationCount,
                valueNetwork=savedValueNetwork,
                policyNetwork=savedPolicyNetwork,
            )
            agent2 = AlphaZeroAgent()
            board = Board()
            replayBuffer = ReplayBuffer(1)
            manager = GameManager(agent1, agent2, board, replayBuffer, verbose=True)
            games3 = manager.playGames(20)
            print(f"Agent has {games3[0] / 20} winrate against random player")

            print(
                f"Current agent won {games[0]} games and saved agent won {games[1]} games."
            )
            if games[0] > int(numGames*2*0.55):
                # save the models
                print("The new model was better. Saving it.")
                self.valueNetwork.save(os.path.join(self.savePath, "valueNetwork.pt"))
                self.policyNetwork.save(os.path.join(self.savePath, "policyNetwork.pt"))
            else:
                print("The new model was not better. Not saving it.")


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
simulationCount = 300
trainAfter = 100
n_epochs = 100
savePath = "models/v1"
trainer = SelfPlayTrainer(
    savePath=savePath, sourcePath=savePath, simulationCount=simulationCount, trainAfter=trainAfter, n_epochs=n_epochs
)
trainer.trainForEpisodes(100)
