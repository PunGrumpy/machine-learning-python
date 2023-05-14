from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


def train_model():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    model, X_test, y_test = train_model()
    save_model(model, "models/model.pkl")


if __name__ == "__main__":
    main()
