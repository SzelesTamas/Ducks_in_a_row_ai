"""When running this file it will train the AlphaZeroAgent against itself."""
from collections import namedtuple, deque
import numpy as np

# we need to implement a data structure to store our data from the games
# we need to implement game manager which can take two agents and make them play against each other a number of times
# we need to implement a function which can take the data from the games and train the neural networks

class ReplayBuffer():
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
        self.experience = namedtuple("Experience", field_names=["state", "targetValue", "targetPolicy"])
        
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
        