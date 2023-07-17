import torch
from torch import nn
import numpy as np


class QNetwork(nn.Module):
    """This is a class for implementing a neural network which inputs a board state, as -1s and 1s. The input layer also takes in the action that was taken, as a x0, y0, x1, y1. The output layer outputs the Q value for the action that was taken."""

    def __init__(
        self,
        modelPath=None,
        inputSize: int = 25 + 4,
        hiddenSizes: list = [25, 25],
        outputSize = 1,        
    ):
        """Initializes the PolicyNetwork class.

        Args:
            modelPath (str, optional): The path to the model file. Defaults to None.
            inputSize (int, optional): The size of the input layer. Defaults to 25 + 4.
            hiddenSizes (list, optional): The sizes of the hidden layers. Defaults to [25, 25].
            outputSize (int, optional): The size of the output layer. Defaults to 1.
        """
        # in the input layer, there are 25 + 4 neurons, one for each square on the board, and each neuron has a value of -1, 0, or 1
        # in the output layer, there are 200 neurons, one for each possible move, and each neuron has a value between 0 and 1
        # the moves are ordered based on the square they start from, and then the square they end on
        # the order of the squares is as follows:
        # 0  1  2  3  4
        # 5  6  7  8  9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # it is basically a row by row flattened version of the board
        # the + 4 is for the 4 neurons that represent the move that was taken

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = torch.flatten
        if modelPath is None:
            layers = []
            layers.append(nn.Linear(inputSize, hiddenSizes[0]))
            layers.append(nn.Sigmoid())
            hiddenSizes.append(outputSize)
            for i in range(0, len(hiddenSizes)-1):
                layers.append(nn.Linear(hiddenSizes[i], hiddenSizes[i+1]))
                layers.append(nn.Sigmoid())
            
            self.model = nn.Sequential(*layers).to(self.device)
        else:
            print("Loading model from " + modelPath)
            self.model = torch.load(modelPath).to(self.device)

    def forward(self, x):
        """This is a function which feeds the input through the neural network. It flattens the input first.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = torch.tensor(x).float()
        x = x.to(self.device)
        return self.model(x).to(self.device)

    def save(self, path):
        """This is a function which saves the model.

        Args:
            path (str): The path to the model file.
        """
        torch.save(self.model, path)
