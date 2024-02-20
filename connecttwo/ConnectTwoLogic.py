'''
Board class.
Board data:
  1=white, -1=black, 0=empty
Squares are stored and manipulated as x index.
x is the column.
'''
import math

class Board():

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = 0

        # Le plateau est vide au début du jeu

    # add [] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def __setitem__(self, index, value): 
        self.pieces[index] = value

    def hasWon(self, color):
        """Boolean for if floor(n/2) adjacent are connected"""
        count = 0
        for x in range(self.n):
            if self[x]==color:
                count += 1
            else:
                count = 0
            if count >= math.floor(self.n/2):
                return True
        return False

    def get_legal_moves(self):
        """Returns all the legal moves for the given player.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        for x in range(self.n):
            # Si le carré est vide alors c'est un coup autorisé
            if self[x]==0:
                newmoves = [x]
                moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self):
        newmoves = self.get_legal_moves()
        if len(newmoves)>0:
            return True
        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; fills with the right color
        (1=white,-1=black)
        """
        if move == self.n+1:
            # Passe le tour
            return
        x = move
        self[x] = color