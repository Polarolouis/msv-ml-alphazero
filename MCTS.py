import logging
import math
import numpy as np


log = logging.getLogger(__name__)


class MCTS:
    def __init__(self, game, nnet, args) -> None:
        self.game = game
        self.nnet = nnet
        
        # Indexes par les tuples (state , action), ou state = str(board), action = index de coup
        self.Q_sa = {}
        self.W_sa = {}
        # Stocke le nombre de visite de l'etat s, a
        self.N_sa = {}

        # Indexes par state = str(board)
        # Stocke le nombre de visite de l'etat s
        self.N_s = {}
        # Le vecteur de proba prior
        self.P_s = {}

        # Stocke les fin de jeu pour s
        self.E_s = {}
        # Les coups valides pour l'etat s
        self.V_s = {}

    def UCB(self, state, action):
        if (state, action) in self.Qsa:
            return self.Q_sa[(state, action)] + self.args.cpuct * self.P_s[state][action] * math.sqrt(self.N_s[state]) / (
                1 + self.N_sa[(state, action)])
        else:
            return self.args.cpuct * self.P_s[state][action] * math.sqrt(self.N_s[state] + 1e-8)


    def search(self, board):
        """Realise le MCTS"""
        # Ici on récupère un string du plateau actuelle
        state = self.game.stringRepresentation(board)

        if state not in self.E_s:
            # Si l'on a pas encore rencontré l'etat state
            self.E_s[state] = self.game.getWinner()
        if self.E_s[state] != 0:
            return -self.E_s[state]
        
        # Extension du noeud et mise en place des priors
        if state not in self.P_s:
            # Noeud feuille de l'arbre
            self.P_s[state], v = self.nnet.predict(board)
            valids = self.game.getValidMoves(board)

            # Etape nécessitant le simulateur pour cacher les coups illicites
            self.P_s[state] = self.P_s[state] * valids  # Cache les coups illicites
            sum_P_s_s = np.sum(self.P_s[state])
            if sum_P_s_s > 0:
                self.P_s[state] /= sum_P_s_s  # renormalize
            else:
                # Si tous les coups sont illicites on triche
                log.error("All valid moves were masked, doing a workaround.")
                self.P_s[state] = self.P_s[state] + valids
                self.P_s[state] /= np.sum(self.P_s[state])

            self.V_s[state] = valids
            self.N_s[state] = 0
            return -v
        
        # Ici on commence la phase de sélection
        valids = self.V_s[state]
        cur_best = -float('inf')
        best_act = -1

        for action in range(self.game.getActionSize()):
            if valids[action]:
                # Si l'action est valide on calcule l'UCB
                u = self.UCB(state=state, action=action)
            if u>cur_best:
                cur_best=u
                best_act = action
        
        # Ici on a recuperer l'action selectionner par l'ucb
        action = best_act
        next_board, next_player = self.game.getNextState(1, action)
        next_board = self.game.getCanonicalForm(next_board, next_player)

        # Entre dans la récursion pour arriver à un noeud terminal et récupérer v
        v = self.search(next_board)

        # Entre dans le backup
        if (state, action) in self.Q_sa:
            # Si on a déjà rencontré ce couple etat action
            self.N_sa[(state, action)] += 1
            # Mise à jour de la valeur totale
            self.W_sa[(state, action)] += v
            # Mise à jour de la valeur moyenne
            self.Q_sa[(state, action)] = self.W_sa[(state, action)] / self.N_sa[(state, action)]
        else:
            # sinon on initialise les valeurs
            self.W_sa[(state, action)] = v
            self.Q_sa[(state, action)] = v
            self.N_sa[(state, action)] = 1

        self.N_s[state] += 1

        return -v

    def getActionProb(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(board)

        s = self.game.stringRepresentation(board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
