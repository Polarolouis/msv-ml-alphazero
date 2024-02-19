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
mini_game = False  # Play in 4 or 8.
human_vs_cpu = True

if mini_game:
    g = ConnectTwoGame(4)
else:
    g = ConnectTwoGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyConnectTwoPlayer(g).play
hp = HumanConnectTwoPlayer(g).play



# nnet players
n1 = NNet(g)
if mini_game:
    n1.load_checkpoint('./pretrained_models/connecttwo/','4conv_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/connecttwo/','8conv_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    if mini_game:
        n2.load_checkpoint('./pretrained_models/connecttwo/','4conv_best.pth.tar')
    else:
        n2.load_checkpoint('./pretrained_models/connecttwo/','8conv_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=ConnectTwoGame.display)

print(arena.playGames(5, verbose=True))
