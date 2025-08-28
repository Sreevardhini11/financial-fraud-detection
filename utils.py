# utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Scale numeric columns safely
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categorical columns safely
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception:
            # fallback: drop problematic categorical
            df.drop(columns=[col], inplace=True)

    return df
