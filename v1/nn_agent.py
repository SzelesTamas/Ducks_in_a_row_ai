from agent import Agent
import random, string
from networks import ValueNetwork
from ducks_in_a_row import Board

class NNAgent(Agent):
    """
    An agent that uses neural networks to play Ducks in a row.
    """
    
    def __init__(self, valueNetwork:ValueNetwork=None, randomness=0.2):
        """Initializes an NNAgent
        
        Args:
            valueNetwork (ValueNetwork, optional): The value network to use. Defaults to None.
            randomness (float, optional): Percentage of moves to be random. Defaults to 0.2.
        """
        
        # random identifier for the agent
        self.id = self.randomID()
        # value network to use
        self.valueNetwork = valueNetwork
        if(self.valueNetwork is None):
            # create a new value network
            self.valueNetwork = ValueNetwork()
        self.randomness = randomness
        self.moveHistory = []
        self.startNewGame()
        
    def startNewGame(self):
        """Add a new row to the move history
        """
        self.moveHistory.append([])
        
    def randomID(self, length=16):
        """Returns a random string of letters and numbers of the given length.
        
        Args:
            length (int, optional): The length of the string to return. Defaults to 16.
        """
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def getMove(self, board:Board, ind:int):
        """Returns the next move to make on the board from the value network, and the value associated with it.

        Args:
            board (Board): The current board state.
            ind (int): The player to make the move for (1 or 2)
        
        Returns:
            tuple: The next move to make as ((x1, y1, x2, y2), value).
        """
        
        # get all possible moves
        validMoves = Board.getValidMoves(board.state, ind)
        moveValues = []
        for move in validMoves:
            # evaluate the state after each move
            nextState = board.move(move[0], move[1], move[2], move[3], inPlace=False).state
            # convert the state to -1s and 1s for the player and opponent
            state = Board.getStateForPlayer(nextState, ind)
            value = self.valueNetwork.getStateValue(state)
            moveValues.append(value)
            
        # choose a random move with probability randomness
        if(random.random() < self.randomness):
            move = random.choice(validMoves)
        else:
            move = validMoves[moveValues.index(max(moveValues))]
        
        # add the move to the move history
        self.moveHistory[-1].append((board, move))
        return move, moveValues[validMoves.index(move)]
