from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .ConnectTwoLogic import Board
import numpy as np

class ConnectTwoGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return ConnectTwoGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n

    def getActionSize(self):
        # return number of actions
        return self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = action
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves()
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x in legalMoves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.hasWon(player):
            return player
        elif b.hasWon(-player):
            return -player
        elif b.has_legal_moves():
            return 0
        return 1e-8

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def stringRepresentation(self, board):
        return board.tobytes()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return 1 if b.areTwoConnected(player) else 0
    
    def getSymmetries(self, board, pi):
        return []

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        print("")
        print((2*n+2)*"_")
        print("|", end="")
        for x in range(n):
            piece = board[x]    # get the piece to print
            print(ConnectTwoGame.square_content[piece], end=" ")
        print("|")

        print((2*n+2)*"‾")
