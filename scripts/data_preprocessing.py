import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    """Load, clean, and preprocess the Boston Housing dataset."""
    # Load dataset
    df = pd.read_csv("data\BostonHousing.csv")

    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Normalize numerical features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Split into features and target
    X = df_scaled.drop(columns=['medv'])
    y = df_scaled['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data
    X_train.to_csv("../data/X_train.csv", index=False)
    X_test.to_csv("../data/X_test.csv", index=False)
    y_train.to_csv("../data/y_train.csv", index=False)
    y_test.to_csv("../data/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test  # Return the processed data

