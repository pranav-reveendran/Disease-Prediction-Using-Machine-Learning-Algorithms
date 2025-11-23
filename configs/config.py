"""
Configuration file for Disease Prediction System
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data files
TRAINING_DATA = PROJECT_ROOT / "Training.csv"
TESTING_DATA = PROJECT_ROOT / "Testing.csv"
PROTOTYPE_DATA = PROJECT_ROOT / "Prototype.csv"

# Model parameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    },
    "decision_tree": {
        "max_depth": 20,
        "min_samples_split": 5,
        "random_state": 42
    },
    "naive_bayes": {
        "var_smoothing": 1e-9
    },
    "neural_network": {
        "hidden_layers": [256, 128, 64],
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "transformer": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 50
    }
}

# Cross-validation
CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging
LOG_LEVEL = "INFO"
