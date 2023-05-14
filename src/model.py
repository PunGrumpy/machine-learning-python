from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import os


def train_model():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def save_model(model, filepath):
    if not os.path.exists("models/"):
        os.makedirs("models/")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def save_data(X_test, y_test):
    if not os.path.exists("data/"):
        os.makedirs("data/")
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)


def main():
    model, X_test, y_test = train_model()
    save_model(model, "models/model.pkl")
    save_data(X_test, y_test)


if __name__ == "__main__":
    main()
