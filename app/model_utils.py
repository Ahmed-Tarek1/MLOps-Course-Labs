"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""
import joblib
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).parent.parent

column_transformer = joblib.load(BASE_DIR / "data" / "column_transformer.joblib")
model = joblib.load(BASE_DIR / "data" / "model.pkl")

FEATURE_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


def predict_churn(features: list) -> int:
    """
    Takes a list of feature values and returns a churn prediction (0 or 1).
    """

    df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    transformed = column_transformer.transform(df)
    
    # Restore feature names expected by the classifier
    transformed_df = pd.DataFrame(
        transformed,
        columns=column_transformer.get_feature_names_out()
    )
    return int(model.predict(transformed_df)[0])

if __name__ == "__main__":
    sample = [600, "France", "Male", 40, 3, 60000.0, 2, 1, 1, 50000.0]
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
