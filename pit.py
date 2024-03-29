import Arena
from MCTS import MCTS

from connecttwo.ConnectTwoGame import ConnectTwoGame
from connecttwo.ConnectTwoPlayers import *
from connecttwo.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

# A modifier pour voir les différentes tailles de grille
gameboard_size = int(input("Enter board size [4,6,8]: ")) # 4, 6, 8
# Sanity check for the values of gameboard size
assert gameboard_size in [4,6,8]
first_player = "rl" # greedy, rl, random
second_player = "human" # human, greedy, rl, random

if gameboard_size not in [4,6,8]:
    raise ValueError("gameboard_size must be 4, 6 or 8")

g = ConnectTwoGame(gameboard_size)

# all players
rp = RandomPlayer(g).play
gp = GreedyConnectTwoPlayer(g).play
hp = HumanConnectTwoPlayer(g).play

if first_player == "greedy":
    n1p = lambda b: gp(b)
elif first_player == "random":
    n1p = lambda b: rp(b)
elif first_player == "rl":
    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/connecttwo/',f'{gameboard_size}best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
else:
    raise ValueError("first_player must be greedy, rl or random")

if second_player == "human":
    n2p = hp
elif second_player == "greedy":
    n2p = lambda b: gp(b)
elif second_player == "random":
    n2p = lambda b: rp(b)
elif second_player == "rl":
    # nnet players
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/connecttwo/',f'{gameboard_size}best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts2 = MCTS(g, n1, args1)
    n2p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
else:
    raise ValueError("second_player must be greedy, rl or random")

player2 = n2p


arena = Arena.Arena(n1p, player2, g, display=ConnectTwoGame.display)

print(arena.playGames(5, verbose=True))
