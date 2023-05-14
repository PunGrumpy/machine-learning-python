from sklearn.datasets import load_diabetes
import pandas as pd
import os


def prepare_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df


def save_data(df, filepath):
    if not os.path.exists("data/"):
        os.makedirs("data/")
    df.to_csv(filepath, index=False)


def main():
    df = prepare_data()
    save_data(df, "data/raw_data.csv")


if __name__ == "__main__":
    main()
