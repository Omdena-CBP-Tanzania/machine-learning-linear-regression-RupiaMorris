import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    X_train = pd.read_csv("../data/X_train.csv")
    y_train = pd.read_csv("../data/y_train.csv")

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "../scripts/trained_model.pkl")

if __name__ == "__main__":
    train_model()
