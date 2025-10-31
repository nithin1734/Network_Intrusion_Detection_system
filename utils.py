# utils.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def preprocess_dataframe(df: pd.DataFrame, label_col: Optional[str] = 'label') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Preprocess dataframe for modeling.
    - If label_col is provided and present in df, returns (X, y).
    - If label_col is None or not present, returns (X, None).
    """
    df = df.copy()

    # Drop obvious non-feature columns if present
    non_features = [c for c in ['src_ip', 'dst_ip', 'timestamp', 'flow_id'] if c in df.columns]
    if label_col and label_col in df.columns:
        non_features = [c for c in non_features if c != label_col]  # keep label_col separate

    # Extract labels if available and requested
    y = None
    if label_col and label_col in df.columns:
        y = df[label_col].apply(lambda v: 0 if str(v).lower() in ['benign', 'normal', '0'] else 1)

    # Drop label and non-feature columns to form features
    drop_cols = non_features + ([label_col] if label_col and label_col in df.columns else [])
    df_feats = df.drop(columns=drop_cols, errors='ignore').fillna(0)

    # Ensure numeric features
    for col in df_feats.columns:
        if not pd.api.types.is_numeric_dtype(df_feats[col]):
            df_feats[col] = pd.to_numeric(df_feats[col], errors='coerce').fillna(0)

    X = df_feats
    return X, y


def load_model(path='model.joblib'):
    import joblib
    obj = joblib.load(path)
    return obj['model'], obj.get('feature_names', None)
