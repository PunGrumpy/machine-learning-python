from sklearn.metrics import mean_squared_error
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.load_and_save import load_model, load_numpy_array


def validate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def main():
    model = load_model("models/model.pkl")
    # load test data
    X_test = load_numpy_array("data_raw/X_test.npy")
    y_test = load_numpy_array("data_raw/y_test.npy")
    mse = validate_model(model, X_test, y_test)
    print(f"Model performance: {mse}")
    if mse > 1.0:  # replace with your own threshold
        raise Exception("Model performance does not meet the threshold")


if __name__ == "__main__":
    main()
