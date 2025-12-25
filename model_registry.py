#!/usr/bin/env python3
"""
Model registry and training utilities.
Provides a flexible framework for training multiple models.
"""

from pathlib import Path
import json
from typing import Dict, Any, Type, Callable
import numpy as np


class ModelRegistry:
    """
    Central registry for all available models.
    Maintains metadata about models and their training parameters.
    """
    
    def __init__(self):
        self.models = {}
    
    def register(self, name: str, player_class: Type, 
                 init_params: Dict[str, Any],
                 has_weights: bool = True,
                 weight_save_fn: Callable = None,
                 weight_load_fn: Callable = None):
        """
        Register a model in the registry.
        
        Args:
            name: Unique model name (e.g., "perceptron", "tableq")
            player_class: The Player class
            init_params: Default init parameters {param: value}
            has_weights: Whether this model learns/saves weights
            weight_save_fn: Function to save weights (or use model.save_weights)
            weight_load_fn: Function to load weights (or use model.load_weights)
        """
        self.models[name] = {
            "class": player_class,
            "init_params": init_params,
            "has_weights": has_weights,
            "save_fn": weight_save_fn,
            "load_fn": weight_load_fn,
        }
    
    def create(self, name: str, player_number: int, **overrides) -> Any:
        """
        Create an instance of a registered model.
        
        Args:
            name: Model name
            player_number: Player number (1 or 2)
            **overrides: Override any init parameters
            
        Returns:
            Player instance
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
        
        model_info = self.models[name]
        params = {**model_info["init_params"], "number": player_number, **overrides}
        return model_info["class"](**params)
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a registered model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
        return self.models[name].copy()
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self.models.keys())
    
    def save_config(self, path: Path):
        """Save registry config to JSON."""
        config = {}
        for name, info in self.models.items():
            config[name] = {
                "class": f"{info['class'].__module__}.{info['class'].__name__}",
                "init_params": info["init_params"],
                "has_weights": info["has_weights"],
            }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _setup_default_models()
    return _registry


def _setup_default_models():
    """Register all built-in models."""
    registry = _registry
    
    # Import player classes from models package
    from models.Myrmidon import Myrmidon
    from models.Perceptron import Perceptron
    from models.SimpleFrequency import SimpleFrequency
    from models.TableQ import TableQ
    from models.RuleBased import RuleBased
    
    # Register Myrmidon (no weights)
    registry.register(
        "myrmidon",
        Myrmidon,
        {"alpha": None, "numSims": 10, "verboseFlag": False},
        has_weights=False
    )
    
    # Register Perceptron (numpy weights)
    registry.register(
        "perceptron",
        Perceptron,
        {"alpha": 0.1, "verboseFlag": False},
        has_weights=True
    )
    
    # Register SimpleFrequency (JSON weights)
    registry.register(
        "simple_frequency",
        SimpleFrequency,
        {"alpha": 0.1, "verboseFlag": False},
        has_weights=True
    )
    
    # Register TableQ (pickle weights)
    registry.register(
        "tableq",
        TableQ,
        {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1, "verboseFlag": False},
        has_weights=True
    )
    
    # Register RuleBased (no weights, deterministic)
    registry.register(
        "rule_based",
        RuleBased,
        {"aggressive": False, "verboseFlag": False},
        has_weights=False
    )
    
    # Variant: aggressive rule-based
    registry.register(
        "rule_based_agg",
        RuleBased,
        {"aggressive": True, "verboseFlag": False},
        has_weights=False
    )


class TrainingConfig:
    """Configuration for training a model."""
    
    def __init__(self, name: str, **kwargs):
        """
        Create training config for a model.
        
        Args:
            name: Model name from registry
            **kwargs: Model-specific parameters and training settings
        """
        self.name = name
        self.model_params = {}
        self.training_params = {
            "rounds": 5000,
            "iterations": 1,
            "benchmark_games": 1000,
            "seed": None,
        }
        
        # Separate model params from training params
        for key, value in kwargs.items():
            if key in self.training_params:
                self.training_params[key] = value
            else:
                self.model_params[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_params": self.model_params,
            "training_params": self.training_params,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create from dictionary."""
        config = cls(d["name"], **d.get("model_params", {}))
        config.training_params.update(d.get("training_params", {}))
        return config
    
    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    # Example usage
    registry = get_registry()
    
    print("Available models:")
    for model_name in registry.list_models():
        info = registry.get_info(model_name)
        print(f"  {model_name}: {info['class'].__name__} (weights: {info['has_weights']})")
    
    # Create an instance
    player = registry.create("perceptron", player_number=1)
    print(f"\nCreated player: {player.name}")
    
    # Create a training config
    config = TrainingConfig(
        "tableq",
        alpha=0.15,
        rounds=3000,
        iterations=2
    )
    print(f"\nTraining config: {config.to_dict()}")
