import pandas as pd


def validate_data(filepath):
    df = pd.read_csv(filepath)
    # implement validation logic here
    # raise exception if validation fails


def main():
    validate_data("data/raw_data.csv")


if __name__ == "__main__":
    main()
