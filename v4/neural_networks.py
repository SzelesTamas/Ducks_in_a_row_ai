"""This is a module for neural networks implemented in pytorch. It contains the PolicyNetwork and ValueNetwork classes."""
import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """This is a class for implementing a neural network which inputs a board state, as -1s and 1s, and outputs a policy vector."""

    def __init__(
        self,
        modelPath=None,
        inputSize: int = 25,
        hiddenSizes: list = [25, 25],
        outputSize: int = 200,
    ):
        """Initializes the PolicyNetwork class.

        Args:
            modelPath (str, optional): The path to the model file. Defaults to None.
            inputSize (int, optional): The size of the input layer. Defaults to 25.
            hiddenSizes (list, optional): The sizes of the hidden layers. Defaults to [25, 25].
            outputSize (int, optional): The size of the output layer. Defaults to 200.
        """
        # in the input layer, there are 25 neurons, one for each square on the board, and each neuron has a value of -1, 0, or 1
        # in the output layer, there are 200 neurons, one for each possible move, and each neuron has a value between 0 and 1
        # the moves are ordered based on the square they start from, and then the square they end on
        # the order of the squares is as follows:
        # 0  1  2  3  4
        # 5  6  7  8  9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # it is basically a row by row flattened version of the board

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = torch.flatten
        if modelPath is None:
            layers = []
            layers.append(nn.Linear(inputSize, hiddenSizes[0]))
            layers.append(nn.Tanh())
            hiddenSizes.append(outputSize)
            for i in range(0, len(hiddenSizes)-1):
                layers.append(nn.Linear(hiddenSizes[i], hiddenSizes[i+1]))
                layers.append(nn.Tanh())
            layers.append(nn.Softmax(dim=1))
            
            self.model = nn.Sequential(*layers).to(self.device)
        else:
            self.model = torch.load(modelPath, map_location=torch.device(self.device)).to(self.device)

    def forward(self, x):
        """This is a function which feeds the input through the neural network. It flattens the input first.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = torch.tensor(x).float()
        x = x.to(self.device)
        x = self.flatten(x, 1)
        return self.model(x)

    def save(self, path):
        """This is a function which saves the model.

        Args:
            path (str): The path to the model file.
        """
        torch.save(self.model, path)


class ValueNetwork(nn.Module):
    """This is a class for implementing a neural network which inputs a board state, as -1s and 1s, and outputs The value of the position for the player with the 1s."""

    def __init__(
        self,
        modelPath=None,
        inputSize: int = 25,
        hiddenSizes: list = [25, 25],
        outputSize: int = 1,
    ):
        """Initializes the ValueNetwork class.

        Args:
            modelPath (str, optional): The path to the model file. Defaults to None.
            inputSize (int, optional): The size of the input layer. Defaults to 25.
            hiddenSizes (list, optional): The sizes of the hidden layers. Defaults to [25, 25].
            outputSize (int, optional): The size of the output layer. Defaults to 1.
        """
        # in the input layer, there are 25 neurons, one for each square on the board, and each neuron has a value of -1, 0, or 1
        # in the output layer, there is 1 neuron, and it has a value between -1 and 1

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = torch.flatten
        if modelPath is None:
            layers = []
            layers.append(nn.Linear(inputSize, hiddenSizes[0]))
            layers.append(nn.Tanh())
            hiddenSizes.append(outputSize)
            for i in range(0, len(hiddenSizes)-1):
                layers.append(nn.Linear(hiddenSizes[i], hiddenSizes[i+1]))
                layers.append(nn.Tanh())
            
            self.model = nn.Sequential(*layers).to(self.device)
        else:
            self.model = torch.load(modelPath, map_location=torch.device(self.device)).to(self.device)

    def forward(self, x):
        """This is a function which feeds the input through the neural network. It flattens the input first.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = torch.tensor(x).float()
        x = x.to(self.device)
        x = self.flatten(x, 1)
        return self.model(x)

    def save(self, path):
        """This is a function which saves the model.

        Args:
            path (str): The path to the model file.
        """
        torch.save(self.model, path)