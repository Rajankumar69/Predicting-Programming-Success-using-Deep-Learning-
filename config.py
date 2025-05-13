# Hyperparameters and paths
import os

DATA_PATH = os.path.join("data", "raw", "programs.csv")
MODEL_PATH = os.path.join("models", "trained_model.h5")
INPUT_SHAPE = 4  # Number of features (budget, team_size, duration, risk_level)
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_STATE = 42