from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import sys
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.load_and_save import save_model, save_numpy_array


def train_model():
    logger.info("Loading and preparing data...")
    X, y = load_diabetes(return_X_y=True)
    logger.debug(f"Data shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor()
    logger.info("Training model...")
    model.fit(X_train, y_train)
    return model, X_test, y_test


def main():
    model, X_test, y_test = train_model()
    logger.info("Saving model and test data...")
    save_model(model, "models/model.pkl")
    save_numpy_array(X_test, "data_raw/X_test.npy")
    save_numpy_array(y_test, "data_raw/y_test.npy")


if __name__ == "__main__":
    logger.add("model.log")
    main()
