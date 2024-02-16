import numpy as np

class ConnectTwoGame:
    def __init__(self) -> None:
        # initialize le plateau
        self.board = np.array([0,0,0,0])
        self.numberOfMove = 0
        self.game_ended = False
        self.winner = 0

    def display(self):
        n = self.board.shape[0]
        print(9*"_")
        print("|",end="")
        for i in range(n):
            if self.board[i] == 1:
                piece = "X"
            elif self.board[i] == -1:
                piece = "O"
            else:
                piece = "."
            print(piece, end="|")
        print("")
        print(9*"‾") 

    
    def isLegalMove(self, move):
        if move >= 0 and self.board[move] == 0:
            return True
        return False

    def playMove(self, player, move):
        # Si l'on ne peut pas jouer on explique pourquoi
        if self.game_ended:
            print("Game Ended")
            return False
        if not player in [-1,1]:
            print("Invalid player")
            return False
        if not self.isLegalMove(move):
            print("Illegal move")
            return False
        # Si l'on peut jouer
        self.numberOfMove += 1
        # On ajoute sur le plateau la valeur correspondant au joueur
        self.board[move] = player
        # On vérifie si après ce coup la partie est finie
        if self.hasGameEnded():
            self.game_ended = True
        return True

    def hasPlayerWon(self, player):
        count = 0
        for idx in range(len(self.board)):
            if self.board[idx] == player:
                count += 1
            else:
                count = 0
            if count >= 2:
                return True
        return False
    
    def hasGameEnded(self):
        """Determines if the game is finished and return a boolean and set the 
        winner if any

        Returns:
            boolean: has the game ended ?
        """
        if self.numberOfMove >= 4:
            return True
        elif self.hasPlayerWon(1):
            self.winner = 1
            return True
        elif self.hasPlayerWon(-1):
            self.winner = -1
            return True
        return False
    
    def getValidMoves(self, board):
        # On initialize le vecteur de taille fixe pour les coups valides
        validMoves = [0] * self.getActionSize()
        for idx in range(self.getActionSize()):
            # On met un indice 1 pour les coups valides et 0 sinon
            validMoves[idx] = 1 if self.isLegalMove(idx) else 0
        return validMoves
    
    def hasLegalMoves(self):
        if any(self.getValidMoves() != 0):
            # Si un coup peut-être joué
            return True
    
    def getWinner(self):
        """Gives out the winner

        Returns:
            int: the int (1,-1) corresponding to the winner, 0 else
        """
        return self.winner
    
    def getCanonicalForm(self, board, player):
        return player * board

    def stringRepresentation(self, board):
        """To provide a unique string matching the board state"""
        return str(board)
    
    def getActionSize(self):
        """Renvoie la taille du plateau car c'est toutes les actions qu'on peut faire"""
        return len(self.board)
    
    def getNextState(self, board, player, action):
        self.playMove(move = action, player = player)
        # On inverse le plateau
        # TODO Mal pensé va surement poser soucis
        #return (-self.board, -player)
        return (-board, -player)