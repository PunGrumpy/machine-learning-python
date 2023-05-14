from sklearn.metrics import mean_squared_error
import pickle
import numpy as np


def validate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def main():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    # load test data
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    mse = validate_model(model, X_test, y_test)
    print(f"Model performance: {mse}")
    if mse > 1.0:  # replace with your own threshold
        raise Exception("Model performance does not meet the threshold")


if __name__ == "__main__":
    main()
