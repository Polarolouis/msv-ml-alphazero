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
gameboard_size = 8 # 4, 6, 8
first_player = "rl" # greedy, rl, random
human_vs_cpu = True

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
    n1.load_checkpoint('./pretrained_models/connecttwo/',f'{gameboard_size}conv_best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
else:
    raise ValueError("first_player must be greedy, rl or random")



if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/connecttwo/',f'{gameboard_size}conv_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

greedy_player = GreedyConnectTwoPlayer(g)
random_player = RandomPlayer(game=g)

gp = lambda b: greedy_player.play(b)
rp = lambda b: random_player.play(b)

#n1p

arena = Arena.Arena(n1p, player2, g, display=ConnectTwoGame.display)

print(arena.playGames(5, verbose=True))
