"""When running this file it will train the AlphaZeroAgent against itself."""
from collections import namedtuple, deque
import numpy as np
from ducks_in_a_row import Board

# we need to implement game manager which can take two agents and make them play against each other a number of times
# we need to implement a function which can take the data from the games and train the neural networks


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
        batch = np.random.choice(self.memory, size=self.batchSize, replace=False)

        states = np.array([exp.state for exp in batch])
        targetValues = np.array([exp.targetValue for exp in batch])
        targetPolicies = np.array([exp.targetPolicy for exp in batch])

        return states, targetValues, targetPolicies

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
        replayBuffer,
        winReward=1.0,
        loseReward=-1.0,
        discountFactor=0.9,
    ):
        """Initializes the GameManager class.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
            replayBuffer (ReplayBuffer): The replay buffer object.
            winReward (float, optional): The reward for winning a game. Defaults to 1.0.
            loseReward (float, optional): The reward for losing a game. Defaults to -1.0.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.replayBuffer = replayBuffer
        self.winReward = winReward
        self.loseReward = loseReward
        self.discountFactor = discountFactor

        self.board = Board()

    def playGame(self):
        """Plays a game between the two agents, calculates the rewards and stores the data in the replay buffer.
        Because we are dealing with sparse rewards we need to calculate the rewards after the game is finished.
        We use a discount factor for providing extra reward for moves that lead to a win.
        Be careful to use a simulationCount over 2304 to avoid creating new roots and losing the data from previous moves
        """
        self.board.startNewGame()
        gameStates = [(self.board.state, self.board.player)]
        while True:
            p = self.board.player
            if self.board.player == 1:
                move, policy = self.agent1.getMove(self.board.state)
            else:
                move, policy = self.agent2.getMove(self.board.state)

            _, nextState, done, winner = self.board.makeMove(Board.moveFromIndex(move))
            gameStates.append((nextState, p, move))
            if done:
                break

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

        for ind, state, player, move in enumerate(gameStates):
            targetValue = (reward1 if player == 1 else reward2) * (
                self.discountFactor ** (len(gameStates) - ind - 1)
            )
            targetPolicy = (
                [child.visits for child in root1.children]
                if player == 1
                else [child.visits for child in root2.children]
            )
            stateForPlayer = Board.getStateForPlayer(state, player)
            self.replayBuffer.addData(stateForPlayer, targetValue, targetPolicy)

            root1 = root1.children[move]
            root2 = root2.children[move]

            if root1 == None:
                print("Error: root1 is None")
                break
            if root2 == None:
                print("Error: root2 is None")
                break

    def playGames(self, numGames):
        """Plays a number of games between the two agents and stores the data from the games in the replay buffer.

        Args:
            numGames (int): The number of games to play.
        """
        for i in range(numGames):
            self.playGame()
