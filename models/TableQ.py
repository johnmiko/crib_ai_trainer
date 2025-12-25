#!/usr/bin/env python3
"""
Tabular Q-Learning Cribbage player.
Uses a table to store Q-values for state-action pairs.
Simple and interpretable learning algorithm.
"""

import numpy as np
from crib_ai_trainer.Player import Player
from itertools import combinations
import pickle
from collections import defaultdict


class TableQ(Player):
    """
    A tabular Q-learner that learns state-action values.
    State is represented by simple hand features.
    """

    def __init__(self, number, alpha=0.1, gamma=0.9, epsilon=0.1, verboseFlag=False):
        """
        Initialize the TableQ player.
        
        Args:
            number: Player number (1 or 2)
            alpha: Learning rate (default 0.1)
            gamma: Discount factor (default 0.9)
            epsilon: Exploration rate (default 0.1)
            verboseFlag: Verbose output (default False)
        """
        super().__init__(number, verboseFlag)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = "TableQ"
        
        # Q-table: state -> action -> Q-value
        # State is a tuple of hand characteristics
        # Action is represented as card index
        self.Q_throw = defaultdict(lambda: defaultdict(float))
        self.Q_play = defaultdict(lambda: defaultdict(float))
        
        # For tracking state-action pairs during learning
        self.lastState = None
        self.lastAction = None
        self.lastReward = 0.0

    def _getThrowState(self):
        """
        Create a state representation for throwing phase.
        Features: (num_pairs, hand_sum_mod_15, num_high_cards)
        """
        if len(self.hand) != 6:
            return (0, 0, 0)
        
        ranks = [c.rank.value for c in self.hand]
        pair_count = sum(1 for i in range(len(ranks)) for j in range(i+1, len(ranks)) if ranks[i] == ranks[j])
        hand_sum = sum(c.value() for c in self.hand) % 15
        high_cards = sum(1 for c in self.hand if c.value() >= 10)
        
        return (pair_count, hand_sum, high_cards)

    def _getPeggingState(self, count, oppCards):
        """
        Create a state representation for pegging phase.
        Features: (count, distance_to_31, opponent_cards_left)
        """
        dist_to_31 = max(0, 31 - count)
        return (count, dist_to_31, oppCards)

    def getThrowCards(self):
        """
        Decide which 4 cards to keep (throws 2).
        Uses epsilon-greedy Q-learning.
        """
        if len(self.hand) != 6:
            return self.hand[:2]
        
        state = self._getThrowState()
        
        # Epsilon-greedy: explore or exploit
        if np.random.random() < self.epsilon:
            # Explore: random combo
            idx = np.random.randint(0, len(list(combinations(range(6), 4))))
            combos = list(combinations(range(6), 4))
            combo_indices = combos[idx]
            best_combo = [self.hand[i] for i in combo_indices]
        else:
            # Exploit: best known Q-value
            best_q = -float('inf')
            best_combo = self.hand[:4]
            
            combos = list(combinations(range(6), 4))
            for combo_indices in combos:
                action = tuple(sorted(combo_indices))
                q_val = self.Q_throw[state].get(action, 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_combo = [self.hand[i] for i in combo_indices]
        
        throw = [c for c in self.hand if c not in best_combo]
        
        # Track for learning
        state = self._getThrowState()
        combo_indices = tuple(sorted([self.hand.index(c) for c in best_combo]))
        self.lastState = ("throw", state, combo_indices)
        self.lastAction = combo_indices
        
        return throw

    def throwCribCards(self, numCards, gameState):
        """Throw numCards cards to crib."""
        return self.getThrowCards()

    def playCard(self, gameState):
        """Play a card during pegging."""
        if not self.playhand:
            return None
        
        legalCards = gameState.get('legalCards', self.playhand)
        count = gameState.get('count', 0)
        
        return self.getPlay(legalCards, count)

    def getPlay(self, legalCards, count):
        """
        Choose which card to play during pegging.
        Uses epsilon-greedy Q-learning with safety constraints.
        """
        if not legalCards:
            return None
        
        # Filter to legal cards (won't bust)
        safe_cards = [c for c in legalCards if count + c.value() <= 31]
        if not safe_cards:
            safe_cards = legalCards
        
        state = self._getPeggingState(count, len(self.playhand))
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # Explore
            best_card = safe_cards[np.random.randint(0, len(safe_cards))]
        else:
            # Exploit: best known Q-value
            best_q = -float('inf')
            best_card = safe_cards[0]
            
            for card in safe_cards:
                action = card.rank.value  # Simple action: rank of card
                q_val = self.Q_play[state].get(action, 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_card = card
        
        # Track for learning
        self.lastState = ("play", state, best_card.rank.value)
        self.lastAction = best_card.rank.value
        
        return best_card

    def learnFromHandScores(self, scores, gameState):
        """
        Learn from hand scoring using Q-learning update rule.
        
        Args:
            scores: List of scores [player1_hand, player2_hand, crib]
            gameState: Current game state dictionary
        """
        my_score = scores[self.number - 1]
        reward = my_score / 100.0  # Normalize reward
        
        if self.lastState and self.lastState[0] == "throw":
            _, state, action = self.lastState
            # Q-learning update: Q(s,a) += alpha * (reward + gamma * max_Q(s') - Q(s,a))
            current_q = self.Q_throw[state].get(action, 0.0)
            # Terminal state, so V(s') = 0
            new_q = current_q + self.alpha * (reward - current_q)
            self.Q_throw[state][action] = new_q

    def learnFromPegging(self, gameState):
        """
        Learn from pegging decisions using Q-learning.
        """
        if self.lastState and self.lastState[0] == "play":
            _, state, action = self.lastState
            
            # Determine reward
            count = gameState.get('count', 0)
            reward = 0.0
            if count == 31:
                reward = 1.0
            elif count == 15:
                reward = 0.5
            elif count <= 30:
                reward = 0.05
            
            current_q = self.Q_play[state].get(action, 0.0)
            new_q = current_q + self.alpha * (reward - current_q)
            self.Q_play[state][action] = new_q

    def explainThrow(self):
        """Explain throwing decision."""
        print(f"TableQ ({self.number}) threw using Q-learning")

    def explainPlay(self):
        """Explain playing decision."""
        print(f"TableQ ({self.number}) played using Q-learning")

    def save_weights(self, path):
        """Save Q-tables to file."""
        weights = {
            "Q_throw": dict(self.Q_throw),
            "Q_play": dict(self.Q_play),
        }
        # Convert defaultdicts to regular dicts for pickling
        weights["Q_throw"] = {k: dict(v) for k, v in self.Q_throw.items()}
        weights["Q_play"] = {k: dict(v) for k, v in self.Q_play.items()}
        
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        """Load Q-tables from file."""
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        
        self.Q_throw = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in weights.get("Q_throw", {}).items()})
        self.Q_play = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in weights.get("Q_play", {}).items()})


if __name__ == "__main__":
    from Arena import Arena
    from Myrmidon import Myrmidon
    
    player1 = TableQ(1, verboseFlag=False)
    player2 = Myrmidon(2, numSims=10, verboseFlag=False)
    
    arena = Arena([player1, player2], repeatDeck=False, verboseFlag=False)
    results = arena.playHands(100)
    
    print(f"TableQ vs Myrmidon (100 hands):")
    print(f"Average point differential: {np.mean(results[2]):.2f}")
