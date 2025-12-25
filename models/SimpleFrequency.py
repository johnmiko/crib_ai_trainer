#!/usr/bin/env python3
"""
Simple frequency-based Cribbage player.
Tracks which cards/combinations tend to score well and learns from them.
"""

import numpy as np
from crib_ai_trainer.Player import Player
from itertools import combinations
import json
from pathlib import Path


class SimpleFrequency(Player):
    """
    A simple learner that tracks card frequencies in scoring hands.
    Makes decisions by choosing cards that appeared in good hands.
    """

    def __init__(self, number, alpha=0.1, verboseFlag=False):
        """
        Initialize the SimpleFrequency player.
        
        Args:
            number: Player number (1 or 2)
            alpha: Learning rate / smoothing factor (default 0.1)
            verboseFlag: Verbose output (default False)
        """
        super().__init__(number, verboseFlag)
        self.alpha = alpha
        self.name = "SimpleFrequency"
        
        # Track: rank -> frequency in high-scoring hands
        self.rankScores = {i: 0.0 for i in range(1, 14)}  # Ace-King
        # Track: suit frequency
        self.suitScores = {i: 0.0 for i in range(1, 5)}   # 4 suits
        # Track: value frequency (1-10)
        self.valueScores = {i: 0.0 for i in range(1, 11)}
        
        # For tracking decisions during learning
        self.lastThrown = None
        self.lastPlayed = None

    def getThrowCards(self):
        """
        Decide which 4 cards to keep (throws 2).
        Uses frequency scores to evaluate combinations.
        """
        if len(self.hand) != 6:
            return self.hand[:2]
        
        best_score = -float('inf')
        best_combo = self.hand[:4]
        
        # Try all combinations of 4 cards to keep
        for combo in combinations(self.hand, 4):
            combo_list = list(combo)
            score = self._scoreCombo(combo_list)
            
            if score > best_score:
                best_score = score
                best_combo = combo_list
        
        # Cards to throw are the 2 not in the best combo
        throw = [c for c in self.hand if c not in best_combo]
        self.lastThrown = throw
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
        Uses frequency scores and safety heuristics.
        """
        if not legalCards:
            return None
        
        best_score = -float('inf')
        best_card = legalCards[0]
        
        for card in legalCards:
            # Frequency score
            freq_score = self.valueScores.get(card.value(), 0.0)
            
            # Safety: avoid busting
            safety = 1.0 if count + card.value() <= 31 else -1.0
            
            # Bonus for 15 or 31
            bonus = 0.0
            if count + card.value() == 31:
                bonus = 2.0
            elif count + card.value() == 15:
                bonus = 1.0
            
            total_score = freq_score + safety + bonus
            
            if total_score > best_score:
                best_score = total_score
                best_card = card
        
        self.lastPlayed = best_card
        return best_card

    def _scoreCombo(self, cards):
        """Score a 4-card combination based on learned frequencies."""
        score = 0.0
        for card in cards:
            score += self.rankScores.get(card.rank.value, 0.0)
            score += self.suitScores.get(card.suit.value, 0.0)
            score += self.valueScores.get(card.value(), 0.0)
        return score

    def learnFromHandScores(self, scores, gameState):
        """
        Learn from hand scoring by updating frequency of cards in good hands.
        
        Args:
            scores: List of scores [player1_hand, player2_hand, crib]
            gameState: Current game state dictionary
        """
        my_score = scores[self.number - 1]
        
        # Only learn from good outcomes
        if my_score > 10:
            # Update based on thrown cards (negative learning for thrown)
            if self.lastThrown:
                for card in self.lastThrown:
                    self.rankScores[card.rank.value] -= self.alpha * (my_score / 100.0)
                    self.suitScores[card.suit.value] -= self.alpha * (my_score / 100.0)
            
            # Update based on kept cards (positive learning)
            kept = [c for c in self.hand if c not in (self.lastThrown or [])]
            for card in kept:
                self.rankScores[card.rank.value] += self.alpha * (my_score / 100.0)
                self.suitScores[card.suit.value] += self.alpha * (my_score / 100.0)

    def learnFromPegging(self, gameState):
        """
        Learn from pegging decisions.
        Update frequency scores based on played cards and outcomes.
        """
        if self.lastPlayed is None:
            return
        
        count = gameState.get('count', 0)
        prev_count = max(0, count - self.lastPlayed.value())
        
        # Reward for good outcomes
        reward = 0.0
        if count == 31:
            reward = 1.0
        elif count == 15:
            reward = 0.5
        elif count < 31:
            reward = 0.1
        
        # Update value score
        self.valueScores[self.lastPlayed.value()] += self.alpha * reward
        
        self.lastPlayed = None

    def explainThrow(self):
        """Explain throwing decision (not implemented)."""
        print(f"SimpleFrequency ({self.number}) threw cards based on learned frequencies")

    def explainPlay(self):
        """Explain playing decision (not implemented)."""
        print(f"SimpleFrequency ({self.number}) played based on frequency and safety")

    def save_weights(self, path):
        """Save learned weights to file."""
        weights = {
            "rankScores": self.rankScores,
            "suitScores": self.suitScores,
            "valueScores": self.valueScores,
        }
        with open(path, 'w') as f:
            json.dump(weights, f, indent=2)

    def load_weights(self, path):
        """Load learned weights from file."""
        with open(path, 'r') as f:
            weights = json.load(f)
        
        # Convert keys back to ints (JSON loads dict keys as strings)
        self.rankScores = {int(k): v for k, v in weights.get("rankScores", {}).items()}
        self.suitScores = {int(k): v for k, v in weights.get("suitScores", {}).items()}
        self.valueScores = {int(k): v for k, v in weights.get("valueScores", {}).items()}


if __name__ == "__main__":
    from crib_ai_trainer.Arena import Arena
    from models.Myrmidon import Myrmidon
    
    player1 = SimpleFrequency(1, verboseFlag=False)
    player2 = Myrmidon(2, numSims=10, verboseFlag=False)
    
    arena = Arena([player1, player2], repeatDeck=False, verboseFlag=False)
    results = arena.playHands(100)
    
    print(f"SimpleFrequency vs Myrmidon (100 hands):")
    print(f"Average point differential: {np.mean(results[2]):.2f}")
