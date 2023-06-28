import torch
from torch import nn
from torch.nn import functional as F
from ducks_in_a_row import Board
import numpy as np
import random

class ValueNetwork():
    """A neural network that predicts the value of a given board state in a range of [-1, 1].
    """
    
    def __init__(self, path:str=None):
        """Initializes the value network.
        """
        super().__init__()
        
        if(path is None):
            # creating a convolutional neural network
            self.model = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 1, padding = 0),
                nn.Tanh(),
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 0),
                nn.Tanh(),
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, padding = 0),
                nn.Tanh(),
                nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, padding = 0),
                nn.Tanh(),
                nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 1, padding = 0),
                nn.Tanh(),
                nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (5, 5), padding = 0),
                nn.Tanh(),
            )
        else:
            self.model = torch.load(path)
        
        self.criteria = nn.MSELoss()
                
    def getStateValue(self, state:np.ndarray):
        """Returns the value of the given board state.
        
        Args:
            state (np.ndarray): The board state to evaluate.
            
        Returns:
            float: The value of the given board state.
        """
        # convert the state to a tensor
        state = torch.from_numpy(state).float()
        # add a dimension for the channel
        state = torch.unsqueeze(state, 0)
        # get the value of the state
        value = self.model(state)
        # return the value
        return value.item()
    
    def train(self, states, values, epochs=1, learningRate=0.001):
        states = torch.from_numpy(states).float()
        # add a dimension for the channel
        states = torch.unsqueeze(states, 1)
        values = torch.from_numpy(values).float()
        
        lossFn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        
        for epoch in range(1, epochs+1):
            # fit the network to the given data
            preds = self.model(states).flatten()
            loss = lossFn(preds, values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print the loss
            print(f"Epoch {epoch}: {loss.item()}")
    
    def save(self, path):
        """Saves the model to the given path.
        
        Args:
            path (str): The path to save the model to.
        """
        torch.save(self.model, path)