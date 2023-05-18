import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.load_and_save import save_numpy_array


def create_dataset():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    save_numpy_array(X_train, "data_raw/X_train.npy")
    save_numpy_array(X_test, "data_raw/X_test.npy")
    save_numpy_array(y_train, "data_raw/y_train.npy")
    save_numpy_array(y_test, "data_raw/y_test.npy")


def create_csv_dataset():
    df = pd.DataFrame(
        data=np.random.rand(100, 5),
        columns=["Feature1", "Feature2", "Feature3", "Feature4", "Target"],
    )
    df.to_csv("data_raw/raw_data.csv", index=False)


def main():
    create_dataset()
    create_csv_dataset()


if __name__ == "__main__":
    main()
