import Arena
import os
import random as rng

from MCTS import MCTS
from connecttwo.ConnectTwoGame import ConnectTwoGame
from connecttwo.ConnectTwoPlayers import *
from connecttwo.pytorch.NNet import NNetWrapper as NNet

import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Setup an arena and store the fraction of wins from the neural network versus the greedy player to plot against iterations
gameboard_size = 8
number_games = 20      

# Game
game = ConnectTwoGame(gameboard_size)

greedy_player = GreedyConnectTwoPlayer(game)
random_player = RandomPlayer(game=game)

# Checkpoint directory
checkpoint_dir = f"./temp/newarch/{gameboard_size}"

# Iterate over the checkpoint files present in ./temp/{gameboard_size}/ to load in the neural network
filelist = []
for checkpoint in os.listdir(checkpoint_dir):
    if checkpoint.endswith('.pth.tar') and len(checkpoint.split('_')) == 2:
        filelist.append(checkpoint)

# Sort file list for the key XX in checkpoint_XX.pth.tar
filelist = sorted(filelist, key=lambda x: int(x.split('_')[1].split('.')[0]))

# Extract iterations number from the selected strings in filelist
iterations = []
for file in filelist:
    iterations.append(int(file.split('_')[1].split('.')[0]))



seed = 1234
rng.seed(seed)
# The data array that will be plotted
data = np.zeros((len(filelist), 8))
data[:,0] = iterations
# For saving data file
iter_max = max(iterations)
saved_data_file = os.path.join(checkpoint_dir, f"{seed}_N{number_games}_iter{iter_max}.npy")
# Check if a file storing data named with boardsize and the seed exists before launching the loop
if os.path.exists(saved_data_file):
    data = np.load(saved_data_file)
else:
    # Iterate over the checkpoints to confront nnet vs greedy_player
    for i in range(len(filelist)):
        nnet = NNet(game)
        nnet.load_checkpoint(folder=checkpoint_dir, filename=filelist[i])
        args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts = MCTS(game, nnet, args)
        n2p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
        print(f"Iteration {i+1}/{len(filelist)} for file {filelist[i]}")
        print("Greedy vs RL")
        # Arena for Greedy player vs NNet
        arena = Arena.Arena(player1=lambda b: greedy_player.play(b), player2=n2p, game = game)
        p1wins, p2wins, draws = arena.playGames(number_games)

        # Greedy player
        data[i,1] = p1wins / number_games 
        # RL player
        data[i,2] = p2wins / number_games
        data[i,3] = draws / number_games

        print("Random vs RL")
        # Arena for Random player vs NNet
        arena = Arena.Arena(player1=lambda b: random_player.play(b), player2=n2p, game = game)
        p1wins, p2wins, draws = arena.playGames(number_games)

        # Random player
        data[i,4] = p1wins / number_games
        # RL player
        data[i,5] = p2wins / number_games
        data[i,6] = draws / number_games
    np.save(saved_data_file, data)


# Plot with legends the fractions of wins for each iterations
plt.style.use(style="ggplot")
plt.figure(figsize=(8, 4))

plt.subplot(1,2,1)
plt.plot(data[:,0], data[:,2], label='RL wins')
plt.plot(data[:,0], data[:,1], label='Greedy wins')
plt.plot(data[:,0], data[:,3], linestyle="--", label='Draws')
plt.xlabel('Iterations')
plt.ylabel('Fraction of wins')
# Set the y axis to be between 0 and 1
plt.ylim(-0.1, 1.1)
plt.legend()

plt.subplot(1,2,2)
plt.plot(data[:,0], data[:,5], label='RL wins')
plt.plot(data[:,0], data[:,4], label='Random wins')
plt.plot(data[:,0], data[:,6], linestyle="--", label='Draws')
plt.xlabel('Iterations')
plt.ylabel('Fraction of wins')
# Set the y axis to be between 0 and 1
plt.ylim(-0.1, 1.1)
plt.legend()
plt.suptitle(f"Fraction of wins for board of size {gameboard_size}\n{number_games} games per iteration")

plt.savefig(os.path.join(checkpoint_dir, f"fraction_wins_{gameboard_size}board.png"))
plt.show()

# Plot for the Random vs NNet


# The MCTS for the neural network
# nmcts = MCTS(game, self.nnet, self.args)


# arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
#                           lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
#             pwins, nwins, draws = arena.playGames(self.args.arenaCompare)