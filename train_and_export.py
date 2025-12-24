#!/usr/bin/env python3
"""
Train AI models and export them to crib_back/models with date stamps.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np

# Cribbage imports
from Arena import Arena

# Player imports
from Myrmidon import Myrmidon
from LinearB import LinearB
from NonLinearB import NonLinearB
from DeepPeg import DeepPeg

# Get today's date for model naming
today = datetime.now().strftime("%Y%m%d")

# Path to crib_back models directory
crib_back_models = Path(__file__).parent.parent / "crib_back" / "models"

print("=" * 80)
print("TRAINING CRIBBAGE AI MODELS")
print(f"Date: {today}")
print("=" * 80)

# ============================================================================
# TRAIN LINEAR B
# ============================================================================
print("\n[1/3] Training LinearB...")
print("-" * 80)

# Create LinearB player with learning parameters
linear_b = LinearB(number=1, alpha=0.5, Lambda=0.9, verboseFlag=False)
opponent = Myrmidon(number=2, numSims=5, verboseFlag=False)

# Train for 100 rounds against Myrmidon
arena = Arena([linear_b, opponent], repeatDeck=False, verboseFlag=False)
print("Training LinearB vs Myrmidon for 100 hands...")
results = arena.playHands(100)
print(f"Training complete. Average total points diff: {np.mean(results[2]):.2f}")

# Save the weights
linear_b_dir = crib_back_models / "linear_b"
throw_weights_path = linear_b_dir / f"throw_weights_{today}.npy"
peg_weights_path = linear_b_dir / f"peg_weights_{today}.npy"

np.save(throw_weights_path, linear_b.throwingWeights)
np.save(peg_weights_path, linear_b.peggingWeights)

print(f"✓ Saved throw weights to: {throw_weights_path}")
print(f"✓ Saved peg weights to: {peg_weights_path}")

# ============================================================================
# TRAIN NON-LINEAR B
# ============================================================================
print("\n[2/3] Training NonLinearB...")
print("-" * 80)

# Create NonLinearB player
non_linear_b = NonLinearB(number=1, alpha=0.3, Lambda=0.7, verboseFlag=False)
opponent = Myrmidon(number=2, numSims=5, verboseFlag=False)

# Train for 100 rounds
arena = Arena([non_linear_b, opponent], repeatDeck=False, verboseFlag=False)
print("Training NonLinearB vs Myrmidon for 100 hands...")
results = arena.playHands(100)
print(f"Training complete. Average total points diff: {np.mean(results[2]):.2f}")

# Save the weights
nlb_throw_weights_path = linear_b_dir / f"nlb_throw_weights_{today}.npy"
nlb_peg_weights_path = linear_b_dir / f"nlb_peg_weights_{today}.npy"

np.save(nlb_throw_weights_path, non_linear_b.throwingWeights)
np.save(nlb_peg_weights_path, non_linear_b.peggingWeights)

print(f"✓ Saved NLB throw weights to: {nlb_throw_weights_path}")
print(f"✓ Saved NLB peg weights to: {nlb_peg_weights_path}")

# ============================================================================
# TRAIN DEEP PEG
# ============================================================================
print("\n[3/3] Training DeepPeg...")
print("-" * 80)

# Create DeepPeg player
deep_peg = DeepPeg(number=1, softmaxFlag=False, saveBrains=False, verbose=False)
opponent = Myrmidon(number=2, numSims=5, verboseFlag=False)

# Train for 100 rounds
print("Training DeepPeg vs Myrmidon for 100 hands...")
for i in range(10):
    arena = Arena([deep_peg, opponent], repeatDeck=False, verboseFlag=False)
    results = arena.playHands(10)
    print(f"  Batch {i+1}/10 complete. Avg diff: {np.mean(results[2]):.2f}")

print("Training complete!")

# Save the brain files
deep_peg_dir = crib_back_models / "deep_peg"

# Import joblib to save the brains
import joblib

peg_brain_path = deep_peg_dir / f"pegging_brain_{today}.pkl"
throw_brain_path = deep_peg_dir / f"throwing_brain_{today}.pkl"

joblib.dump(deep_peg.peggingBrain, peg_brain_path)
joblib.dump(deep_peg.throwingBrain, throw_brain_path)

print(f"✓ Saved pegging brain to: {peg_brain_path}")
print(f"✓ Saved throwing brain to: {throw_brain_path}")

# ============================================================================
# CREATE MYRMIDON CONFIG
# ============================================================================
print("\n[4/4] Creating Myrmidon config...")
print("-" * 80)

myrmidon_dir = crib_back_models / "myrmidon"
config_path = myrmidon_dir / "config.txt"

with open(config_path, 'w') as f:
    f.write(f"# Myrmidon Configuration\n")
    f.write(f"# Created: {today}\n")
    f.write(f"# Myrmidon is heuristic-based and requires no trained parameters\n")
    f.write(f"# Default simulations: 5\n")

print(f"✓ Created config at: {config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModels saved to: {crib_back_models}")
print(f"\nModel files created:")
print(f"  LinearB:")
print(f"    - throw_weights_{today}.npy")
print(f"    - peg_weights_{today}.npy")
print(f"    - nlb_throw_weights_{today}.npy")
print(f"    - nlb_peg_weights_{today}.npy")
print(f"  DeepPeg:")
print(f"    - pegging_brain_{today}.pkl")
print(f"    - throwing_brain_{today}.pkl")
print(f"  Myrmidon:")
print(f"    - config.txt")
print("\nNext steps:")
print("  1. Update crib_back/cribbage/opponents.py to load these models")
print("  2. Add new opponent types to the registry")
print("  3. Test the trained opponents in the game")
print("=" * 80)
