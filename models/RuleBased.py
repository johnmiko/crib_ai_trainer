#!/usr/bin/env python3
"""
Rule-based Cribbage player with heuristic decision-making.
Uses hand-crafted rules to estimate card value and make decisions.
Simple, interpretable, and serves as a good baseline.
"""

import numpy as np
from crib_ai_trainer.Player import Player
from itertools import combinations


class RuleBased(Player):
    """
    A rule-based player that uses heuristic scoring for decisions.
    Evaluates cards based on cribbage scoring rules and strategy.
    """

    def __init__(self, number, aggressive=False, verboseFlag=False):
        """
        Initialize the RuleBased player.
        
        Args:
            number: Player number (1 or 2)
            aggressive: If True, take more risks (default False = conservative)
            verboseFlag: Verbose output (default False)
        """
        super().__init__(number, verboseFlag)
        self.aggressive = aggressive
        self.name = "RuleBased_Agg" if aggressive else "RuleBased"

    def getThrowCards(self):
        """
        Decide which 4 cards to keep (throws 2).
        Uses heuristic scoring based on cribbage principles.
        """
        if len(self.hand) != 6:
            return self.hand[:2]
        
        best_score = -float('inf')
        best_combo = self.hand[:4]
        
        # Try all combinations of 4 cards to keep
        for combo in combinations(self.hand, 4):
            combo_list = list(combo)
            score = self._scoreHandForKeeping(combo_list)
            
            if score > best_score:
                best_score = score
                best_combo = combo_list
        
        # Cards to throw are the 2 not in the best combo
        throw = [c for c in self.hand if c not in best_combo]
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
        Uses heuristic scoring with safety priorities.
        """
        if not legalCards:
            return None
        
        best_score = -float('inf')
        best_card = legalCards[0]
        
        for card in legalCards:
            score = self._scorePeggingPlay(card, count, len(self.playhand))
            
            if score > best_score:
                best_score = score
                best_card = card
        
        return best_card

    def _scoreHandForKeeping(self, cards):
        """
        Score a 4-card hand kept for scoring phase.
        Evaluates based on pairing, fifteening, run potential, etc.
        """
        score = 0.0
        
        # 1. Pairs and near-pairs
        ranks = [c.rank.value for c in cards]
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                if ranks[i] == ranks[j]:
                    score += 20.0  # Exact pair
                elif abs(ranks[i] - ranks[j]) == 1:
                    score += 2.0   # Close to pair (better for runs)
        
        # 2. Fifteen potential
        values = [c.value() for c in cards]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                for k in range(j + 1, len(values)):
                    if values[i] + values[j] + values[k] == 15:
                        score += 8.0
                    elif values[i] + values[j] + values[k] == 15 - 1:
                        score += 1.0  # Close to 15
        
        # 3. Run potential (consecutive ranks)
        sorted_ranks = sorted(ranks)
        run_length = 1
        for i in range(len(sorted_ranks) - 1):
            if sorted_ranks[i + 1] - sorted_ranks[i] == 1:
                run_length += 1
            else:
                if run_length >= 3:
                    score += (run_length * 5.0)
                run_length = 1
        if run_length >= 3:
            score += (run_length * 5.0)
        
        # 4. Avoid very low value combos
        total_value = sum(values)
        if total_value < 8:
            score -= 5.0
        
        # 5. Prefer variety of values (less repetition)
        unique_values = len(set(values))
        score += unique_values
        
        return score

    def _scorePeggingPlay(self, card, count, playhand_size):
        """
        Score a card for playing during pegging phase.
        Prioritizes scoring and safety.
        """
        score = 0.0
        new_count = count + card.value()
        
        # 1. Exact scoring: 31 is best
        if new_count == 31:
            score += 100.0
        elif new_count == 15:
            score += 30.0
        
        # 2. Safety: avoid busting
        if new_count > 31:
            score -= 1000.0  # Avoid at all costs
        else:
            # Prefer getting closer to 31, but not too close
            dist_to_31 = 31 - new_count
            if dist_to_31 <= 3:
                score += 5.0  # Good position
            elif dist_to_31 <= 10:
                score += 2.0  # Reasonable position
        
        # 3. Card value considerations
        if card.value() < 5:
            score += 3.0  # Low cards are safer
        
        # 4. Opponent strength: if many cards left for opponent, be cautious
        if self.aggressive:
            # Aggressive: play higher cards
            score += card.value() * 0.5
        else:
            # Conservative: play lower cards
            score += (10 - card.value()) * 0.5
        
        # 5. Proximity to go
        if count < 10:
            score += 1.0  # More safe zone, more flexible
        
        return score

    def learnFromHandScores(self, scores, gameState):
        """
        Hand scores are known; could update heuristics here.
        For now, this is a rule-based player so no learning.
        
        Args:
            scores: List of scores [player1_hand, player2_hand, crib]
            gameState: Current game state dictionary
        """
        pass

    def learnFromPegging(self, gameState):
        """
        No learning for rule-based player.
        """
        pass

    def explainThrow(self):
        """Explain throwing decision."""
        style = "aggressively" if self.aggressive else "conservatively"
        print(f"RuleBased ({self.number}) threw {style} based on heuristics")

    def explainPlay(self):
        """Explain playing decision."""
        style = "aggressively" if self.aggressive else "conservatively"
        print(f"RuleBased ({self.number}) played {style} based on heuristics")


if __name__ == "__main__":
    from crib_ai_trainer.Arena import Arena
    from models.Myrmidon import Myrmidon
    
    player1 = RuleBased(1, aggressive=False, verboseFlag=False)
    player2 = Myrmidon(2, numSims=10, verboseFlag=False)
    
    arena = Arena([player1, player2], repeatDeck=False, verboseFlag=False)
    results = arena.playHands(100)
    
    print(f"RuleBased vs Myrmidon (100 hands):")
    print(f"Average point differential: {np.mean(results[2]):.2f}")
