from ducks_in_a_row import Board
from nn_agent import NNAgent
from networks import ValueNetwork
import numpy as np
import random
import sys
from datetime import datetime

def playGame(players:list, maxMoves=100, winReward:float=1.0, loseReward:float=-1.0, discountFactor:float=0.9):
    """Plays a game of Ducks in a row between the given players and returns the outcome and the states of the games when each player decided.
    
    Args:
        players (list): The players to play with.
        maxMoves (int, optional): The maximum number of moves to play. Defaults to 100.
        winReward (float, optional): The reward for winning. Defaults to 1.0.
        loseReward (float, optional): The reward for losing. Defaults to -1.0.
        discountFactor (float, optional): The discount factor for the rewards. Defaults to 0.9.
        
    Returns:
        tuple: The index+1 of the winner of the game. 0 if the game was a draw. The states of the game when each player decided with the values associated with it
    """
    
    winner = 0
    steps = [[], []]
    board = Board()
    turn = 0
    for i in range(maxMoves):
        move, value = players[turn].getMove(board, turn+1)
        board.move(move[0], move[1], move[2], move[3])
        steps[turn].append(Board.getStateForPlayer(board.state, turn+1))
        turn = (turn + 1) % 2
        
        outcome = board.gameOver()
        if(outcome != 0):
            winner = outcome
            break
    
    if(winner != 0):
        # apply the final state for both players
        steps[turn].append(Board.getStateForPlayer(board.state, turn+1))
        steps[(turn+1)%2].append(Board.getStateForPlayer(board.state, (turn+1)%2+1))
        
        rewards = [[], []]
        # swap the winner to the first player
        if(winner == 2):
            steps = steps[::-1]
        # calculate the rewards
        for i in range(len(steps[0])):
            rewards[0].append(winReward*(discountFactor**(len(steps[0])-1-i)))
        for i in range(len(steps[1])):
            rewards[1].append(loseReward*(discountFactor**(len(steps[1])-1-i)))
        
        steps = steps[0] + steps[1]
        rewards = rewards[0] + rewards[1]
        return winner, (steps, rewards)
    else:
        return 0, []
    
def trainAgainstRandom(path, n_games=100, maxMoves=100, trainFrequency=10, epochs=10, learningRate=0.01, winReward=1.0, loseReward=-1.0, discountFactor=0.9, randomness=0.2):
    """Tests the given network by playing the given number of games against a random agent.
    
    Args:
        network (ValueNetwork): The network to test.
        n_games (int, optional): The number of games to play. Defaults to 100.
        maxMoves (int, optional): The maximum number of moves to play. Defaults to 100.
        trainFrequency (int, optional): The number of games to play before training the network. Defaults to 10.
        epochs (int, optional): The number of epochs to train the network for. Defaults to 10.
    """
    
    network = ValueNetwork(path=path)
    # create a list of players
    players = [NNAgent(valueNetwork = network, randomness=randomness), NNAgent()]
    trainData = [[], []]
    results = [0, 0, 0]
    if(path == None):
        path = "models/valueNetwork_" + ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(4)) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5" 
    
    for i in range(1, n_games+1):
        
        starts = random.random() < 0.5
        if(not starts):
            winner, decisions = playGame(players, maxMoves=maxMoves, winReward=winReward, loseReward=loseReward, discountFactor=discountFactor)
            results[winner] += 1
        else:
            winner, decisions = playGame(players[::-1], maxMoves=maxMoves, winReward=winReward, loseReward=loseReward, discountFactor=discountFactor)
            if(winner == 0):
                results[0] += 1
            else:
                results[(winner-1)%2+1] += 1
            
        if(winner != 0):
            trainData[0] += decisions[0]
            trainData[1] += decisions[1]
        
        print(f"Game {i} complete. Winner: {winner} Steps taken: {99 if len(decisions) == 0 else (len(decisions[1]))}", end="\r")
            
        if(trainFrequency != 0 and i % trainFrequency == 0 and len(trainData[0]) != 0):
            print()
            # train the network
            print(f"Training network with {len(trainData[0])} states. Average reward: {sum(trainData[1])/len(trainData[1])}")
            print(f"Wins: {results[1]} Losses: {results[2]} Draws: {results[0]} Winrate: {results[1]/(results[1]+results[2]) if results[1]+results[2] != 0 else 0}")
            # train the network on the states
            network.train(np.array(trainData[0]), np.array(trainData[1]), epochs=epochs, learningRate=learningRate)
            
            print("Training complete. Saving network...")
            network.save(path)
            print("Network saved.")
            trainData = [[], []]
    print()
    
def trainNetwork(path, n_games=100, maxMoves=100, trainFrequency=10, epochs=10, learningRate=0.01, winReward=1.0, loseReward=-1.0, discountFactor=0.9, randomness=0.2):
    """Tests the given network by playing the given number of games against a random agent.
    
    Args:
        network (ValueNetwork): The network to test.
        n_games (int, optional): The number of games to play. Defaults to 100.
        maxMoves (int, optional): The maximum number of moves to play. Defaults to 100.
        trainFrequency (int, optional): The number of games to play before training the network. Defaults to 10.
        epochs (int, optional): The number of epochs to train the network for. Defaults to 10.
    """
    
    network = ValueNetwork(path=path)
    # create a list of players
    players = [NNAgent(valueNetwork = network, randomness=randomness), NNAgent(valueNetwork=network, randomness=randomness)]
    trainData = [[], []]
    if(path == None):
        path = "models/valueNetwork_" + ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(4)) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5" 
    for i in range(1, n_games+1):
        
        starts = random.random() < 0.5
        if(not starts):
            winner, decisions = playGame(players, maxMoves=maxMoves, winReward=winReward, loseReward=loseReward, discountFactor=discountFactor)
        else:
            winner, decisions = playGame(players[::-1], maxMoves=maxMoves, winReward=winReward, loseReward=loseReward, discountFactor=discountFactor)
            
        if(winner != 0):
            trainData[0] += decisions[0]
            trainData[1] += decisions[1]
        
        print(f"Game {i} complete. Winner: {winner} Steps taken: {99 if len(decisions) == 0 else (len(decisions[1]))}", end="\r")
            
        if(trainFrequency != 0 and i % trainFrequency == 0 and len(trainData[0]) != 0):
            print()
            # train the network
            print(f"Training network with {len(trainData[0])} states. Average reward: {sum(trainData[1])/len(trainData[1])}")
            # train the network on the states
            network.train(np.array(trainData[0]), np.array(trainData[1]), epochs=epochs, learningRate=learningRate)
            
            print("Training complete. Saving network...")
            network.save(path)
            print("Network saved.")
            trainData = [[], []]
    print()

def testNetwork(path, n_games=100, maxMoves=100):
    """Tests the given network by playing the given number of games against a random agent.
    
    Args:
        path (str): The path to the network to test.
        n_games (int, optional): The number of games to play. Defaults to 100.
        maxMoves (int, optional): The maximum number of moves to play. Defaults to 100.
        
    Returns:
        list: []
    """
    if(n_games == 0):
        return
    
    players = [NNAgent(valueNetwork = ValueNetwork(path=path), randomness=0.0), NNAgent()]
    results = [0, 0, 0]
    for i in range(1, n_games+1):
        if(random.random() < 0.5):
            winner, decisions = playGame(players, maxMoves=maxMoves)
            results[winner] += 1
        else:
            winner, decisions = playGame(players[::-1], maxMoves=maxMoves)
            if(winner == 0):
                results[winner] += 1
            else:
                results[winner%2+1] += 1
    
    print("Test Results")
    print(f"Wins: {results[1]}, Losses: {results[2]}, Draws: {results[0]}")
    print(f"Win rate: {results[1]/n_games}, Loss rate: {results[2]/n_games}, Draw rate: {results[0]/n_games}")
    return results

modelPath = "models/valueNetwork_haec_20230623-205515.h5"
trainAgainstRandom(modelPath, n_games=10000, maxMoves=200, trainFrequency=20, epochs=3)

# testing the network against a random player
testNetwork(modelPath, n_games=100, maxMoves=50)



