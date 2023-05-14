import pandas as pd
import os
import sys


def validate_data():
    dir_path = "data_raw"
    file_path = os.path.join(dir_path, "raw_data.csv")
    df = pd.read_csv(file_path)
    # implement validation logic here
    if df.shape[0] < 100:
        raise Exception("Dataset size is too small")
    # raise exception if validation fails
    if df.shape[1] != 5:
        raise Exception("Dataset does not have the right number of columns")


def main():
    validate_data()


if __name__ == "__main__":
    main()
