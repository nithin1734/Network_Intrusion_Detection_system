# train_model.py (robust to small datasets)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils import preprocess_dataframe

DATA_PATH = 'sample_data/sample_flow.csv'  # replace with your dataset path
MODEL_PATH = 'model.joblib'

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_dataframe(df)

    # Basic checks for stratification viability
    n_samples = len(y)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    # If dataset is large enough and each class has at least 2 samples, do stratified split
    try:
        if n_samples >= 10 and min_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            print('Accuracy:', accuracy_score(y_test, preds))
            print(classification_report(y_test, preds))

        else:
            # Too small for a reliable split — train on entire dataset
            print(f"Dataset too small for a stratified split (n_samples={n_samples}, min_class_count={min_count}).")
            print("Training on the entire dataset (no train/test split).")
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X, y)
            # Skip printing metrics because we didn't hold out a test set.

        # Save model and feature names
        joblib.dump({'model': clf, 'feature_names': list(X.columns)}, MODEL_PATH)
        print(f'Model saved to {MODEL_PATH}')

    except Exception as e:
        print("Training failed:", str(e))
        raise

if __name__ == '__main__':
    main()
