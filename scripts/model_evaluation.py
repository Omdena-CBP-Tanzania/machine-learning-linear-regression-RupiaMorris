import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    X_test = pd.read_csv("../data/X_test.csv")
    y_test = pd.read_csv("../data/y_test.csv")

    model = joblib.load("../scripts/trained_model.pkl")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

if __name__ == "__main__":
    evaluate_model()
