"""
Preprocessing pipeline for Customer Churn Prediction.
Handles encoding, scaling, imputation and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

CATEGORICAL_BINARY = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn"
]

CATEGORICAL_MULTI = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "NumSupportTickets"
]

DROP_COLS = ["customerID"]


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """End-to-end preprocessing transformer for churn data."""

    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
        self.scaler = StandardScaler()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop ID columns
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")

        # Convert TotalCharges to numeric (sometimes has spaces)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Fill missing numeric
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        return df

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features."""
        df = df.copy()
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["ChargesPerMonth"] = df["MonthlyCharges"]
            df["TenureGroup"] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 36, 60, 100],
                labels=["0-12m", "13-24m", "25-36m", "37-60m", "60m+"]
            ).astype(str)
            df["HighValueCustomer"] = (
                (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75))
                & (df["tenure"] > 24)
            ).astype(int)

        if "NumSupportTickets" in df.columns:
            df["HighSupportUsage"] = (df["NumSupportTickets"] >= 3).astype(int)

        return df

    def fit(self, df: pd.DataFrame, y=None):
        df = self._clean(df)
        df = self._feature_engineer(df)

        # Encode binary cats
        for col in CATEGORICAL_BINARY:
            if col in df.columns and col != "Churn":
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le

        # One-hot encode multi cats (fit by storing dummies columns)
        for col in CATEGORICAL_MULTI + ["TenureGroup"]:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le

        # Transform to get feature columns
        transformed = self._transform_internal(df)
        self.feature_columns = transformed.columns.tolist()

        # Fit scaler
        numeric_cols = [c for c in NUMERIC_FEATURES + ["ChargesPerMonth"] if c in transformed.columns]
        if numeric_cols:
            self.scaler.fit(transformed[numeric_cols])
            self._numeric_cols = numeric_cols
        else:
            self._numeric_cols = []

        return self

    def _transform_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Encode binary
        for col in CATEGORICAL_BINARY:
            if col in df.columns and col != "Churn":
                le = self.label_encoders.get(col)
                if le:
                    df[col] = le.transform(df[col].astype(str))

        # One-hot multi cats
        ohe_cols = [c for c in CATEGORICAL_MULTI + ["TenureGroup"] if c in df.columns]
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

        # Drop target if present
        df.drop(columns=["Churn"], inplace=True, errors="ignore")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean(df)
        df = self._feature_engineer(df)
        df = self._transform_internal(df)

        # Align columns to training schema
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]

        # Scale numeric
        if self._numeric_cols:
            present = [c for c in self._numeric_cols if c in df.columns]
            if present:
                df[present] = self.scaler.transform(df[present])

        # ── dtype safety ──────────────────────────────────────────────────
        # pd.get_dummies() produces bool columns in pandas ≥ 1.5.
        # sklearn 1.8 internal validation can raise AttributeError on certain
        # model attributes (e.g. multi_class, removed in 1.8) when it
        # encounters non-float input during dtype coercion.
        # Casting to float64 here is the single safest fix: it eliminates
        # bool, int8, uint8, and any object columns in one step, and ensures
        # NaN that survived imputation becomes 0.0 via fillna.
        df = df.fillna(0).astype(np.float64)

        return df

    def get_feature_names(self):
        return self.feature_columns or []


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Extract and encode target variable."""
    return (df["Churn"].map({"Yes": 1, "No": 0})
            .fillna(0).astype(int))
