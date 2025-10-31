# app.py - final Streamlit app for AI NIDS (supervised + unsupervised)
import io
import os
import csv
import statistics
import streamlit as st
import pandas as pd
import numpy as np
import shap
from utils import preprocess_dataframe, load_model

from sklearn.ensemble import IsolationForest

st.set_page_config(page_title='AI NIDS', layout='wide')


@st.cache_resource
def get_model(path='model.joblib'):
    """
    Load model from path. If missing, try to safely train using train_model.main().
    """
    if not os.path.exists(path):
        st.warning("Model not found — attempting to train a new model (safe mode)...")
        try:
            import train_model
            train_model.main()
        except Exception as e:
            st.error(f"Automatic training failed: {e}")
            raise
    return load_model(path)


def read_uploaded_table(uploaded, encoding='auto', delim='auto', header_row='auto'):
    """
    Very robust reader for uploaded CSV/XLSX/binary-like files.
    - Tries Excel if file is an xlsx/zip blob (PK..).
    - Checks gzip, zip magic bytes.
    - Tries multiple encodings.
    - If pandas can't parse, falls back to a heuristic tokenizer using regex.
    - Returns pd.DataFrame or raises ValueError with diagnostics.
    """
    import io, re, struct
    if uploaded is None:
        raise ValueError("No file provided")

    name = getattr(uploaded, "name", "")
    raw = uploaded.getvalue()  # bytes

    if not raw:
        raise ValueError("Uploaded file is empty")

    # Show sample for diagnostics in sidebar
    try:
        sample_for_display = raw[:2048].decode('latin1', errors='replace')
    except Exception:
        sample_for_display = str(raw[:2048])
    st.sidebar.subheader("File sample (first 2KB, latin1)")
    st.sidebar.code(sample_for_display[:1000])

    # Detect common binary container types by magic bytes
    MAGIC = raw[:4]
    # XLSX/ZIP starts with PK\x03\x04
    if MAGIC.startswith(b'PK'):
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            st.sidebar.info(f"Detected ZIP/PK (maybe xlsx) but read_excel failed: {e}")

    # gzip magic
    if raw[:2] == b'\x1f\x8b':
        try:
            import gzip
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                decompressed = gz.read()
            # try excel first, then csv
            try:
                return pd.read_excel(io.BytesIO(decompressed))
            except Exception:
                try:
                    txt = decompressed.decode('utf-8', errors='replace')
                    return pd.read_csv(io.StringIO(txt), sep=None, engine='python', on_bad_lines='skip')
                except Exception as e:
                    st.sidebar.info(f"Gzip decoded but parsing failed: {e}")
        except Exception as e:
            st.sidebar.info(f"Gzip handling failed: {e}")

    # If extension .xls/.xlsx try excel first
    if name.lower().endswith(('.xls', '.xlsx')):
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            st.sidebar.info(f"read_excel failed: {e}")

    # Determine header param
    if header_row == 'auto':
        header_param = 0
    else:
        try:
            hr = int(str(header_row).strip())
            header_param = None if hr == -1 else hr
        except Exception:
            header_param = 0

    # Attempt pandas' inference (sep=None) with a few encodings
    enc_list = [encoding] if encoding != 'auto' else ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']
    last_exc = None
    for enc in enc_list:
        try:
            # try read as bytes buffer using pandas inference
            buf = io.BytesIO(raw)
            df = pd.read_csv(buf, sep=None, engine='python', header=header_param, encoding=enc, on_bad_lines='skip')
            if df is not None and df.shape[1] > 0:
                st.sidebar.info(f"Parsed using pandas infer with encoding={enc}")
                return df
        except Exception as e:
            last_exc = e
            st.sidebar.info(f"pandas infer with encoding={enc} failed: {e}")
            continue

    # Try grid of delimiters and encodings (on whole decoded text)
    delims_try = [',', ';', '\t', '|', ' ']
    best = None  # (ncols, nrows, enc, delim, df)
    for enc in enc_list:
        try:
            txt = raw.decode(enc, errors='replace')
        except Exception as e:
            last_exc = e
            continue
        for sep in delims_try:
            try:
                df_try = pd.read_csv(io.StringIO(txt), sep=sep, header=header_param, engine='python', on_bad_lines='skip')
                ncols, nrows = df_try.shape[1], df_try.shape[0]
                # pick the parse with most columns (and then more rows)
                if ncols > 1 and (best is None or (ncols, nrows) > (best[0], best[1])):
                    best = (ncols, nrows, enc, sep, df_try)
            except Exception as e:
                last_exc = e
                continue

    if best:
        st.sidebar.info(f"Best grid parse: encoding={best[2]}, delimiter={repr(best[3])}, columns={best[0]}")
        return best[4]

    # FINAL HEURISTIC TOKENIZER: split lines and tokenize by regex -> choose most consistent column count
    try:
        # decode with permissive utf-8 (replace)
        text = raw.decode('utf-8', errors='replace')
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("No non-empty lines found in file after decoding.")

        # Candidate tokenizers
        tokenizers = [
            re.compile(r'[,\t;|]+'),        # common separators
            re.compile(r'\s{2,}'),         # runs of multiple spaces
            re.compile(r'\s*\|\s*'),       # pipe (tight)
        ]

        best_tok = None  # (std_dev, mean_cols, delim_desc, tokens, parsed_table)
        for tok_re in tokenizers:
            token_counts = []
            parsed_rows = []
            for ln in lines[:1000]:  # sample up to 1000 lines
                parts = [p.strip() for p in tok_re.split(ln)]
                token_counts.append(len(parts))
                parsed_rows.append(parts)
            # compute variability and mean
            if not token_counts:
                continue
            mean_cols = sum(token_counts) / len(token_counts)
            try:
                std = statistics.pstdev(token_counts)
            except Exception:
                std = float('inf')
            # Only consider tokenizers that produce >1 column on avg
            if mean_cols > 1:
                # create a DataFrame padding shorter rows with None
                max_cols = max(token_counts)
                padded = [row + [None] * (max_cols - len(row)) for row in parsed_rows]
                parsed_df = pd.DataFrame(padded)
                score = (std, -mean_cols)  # lower std and higher mean_cols preferred
                if best_tok is None or score < best_tok[0]:
                    best_tok = (score, mean_cols, tok_re.pattern, parsed_df)

        if best_tok is not None:
            st.sidebar.info(f"Heuristic tokenizer used: pattern {best_tok[2]}, mean_cols={best_tok[1]:.2f}")
            df_out = best_tok[3]
            # If header_param != None, treat first row as header where possible
            if header_param is not None and df_out.shape[0] > header_param:
                try:
                    df_out.columns = df_out.iloc[header_param].astype(str).tolist()
                    df_out = df_out.drop(index=header_param).reset_index(drop=True)
                except Exception:
                    pass
            return df_out
    except Exception as e:
        last_exc = e

    # Nothing worked: raise with diagnostics
    raise ValueError(
        "Could not parse file into table columns. Possible causes:\n"
        "- The file uses a non-standard delimiter. Try selecting a delimiter in the sidebar (',', ';', or '\\t').\n"
        "- The file is not a CSV/Excel (e.g., binary or formatted log). Open the file in a text editor to inspect.\n"
        "- The file has metadata lines or no header; set Header row to -1 or the appropriate index.\n\n"
        f"Last parsing exception: {last_exc}\n\n"
        "Check the 'File sample (first 2KB)' in the sidebar to decide encoding/delimiter/header."
    )


def main():
    st.title('AI-powered Network Intrusion Detection (NIDS)')

    st.sidebar.header("Mode & Model")
    mode = st.sidebar.selectbox("Operation mode", ["Supervised (use label column)", "Unsupervised (anomaly detection)"])

    # If supervised, attempt to load model (auto-train if missing)
    model = None
    feature_names = None
    if mode.startswith("Supervised"):
        try:
            model, feature_names = get_model('model.joblib')
        except Exception as e:
            st.error(f"Could not load/create model: {e}")
            return

    st.sidebar.markdown("---")
    st.sidebar.write("If using supervised mode, you can specify a label column name or choose a label column index after upload (useful if headers are garbled).")

    # --- Upload controls & parsing overrides ---
    st.sidebar.header("Upload options (override if parsing fails)")
    encoding_choice = st.sidebar.selectbox("Encoding", ['auto', 'utf-8', 'cp1252', 'iso-8859-1', 'latin1'])
    delim_choice = st.sidebar.selectbox("Delimiter", ['auto', ',', ';', '\\t', '|'])
    header_choice = st.sidebar.text_input("Header row (use -1 if no header) or 'auto'", value="auto")

    uploaded = st.file_uploader('Upload network flow file (CSV or Excel)', type=['csv', 'xls', 'xlsx'])
    if uploaded is None:
        st.info('Upload a CSV/Excel file to get predictions.')
        return

    # Read file using robust helper
    try:
        df = read_uploaded_table(uploaded, encoding=encoding_choice, delim=delim_choice, header_row=header_choice)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        return

    # If file has no header or columns look numeric-only, rename columns to col_0...
    if str(header_choice).strip() == '-1' or (list(df.columns) == [0] and df.shape[1] > 1 and not any(isinstance(c, str) for c in df.columns)):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        st.info("No header detected: columns renamed to col_0, col_1, ...")

    st.subheader('Preview of uploaded data')
    st.dataframe(df.head())

    # Show detected columns with indices in sidebar
    cols_show = list(df.columns)
    st.sidebar.subheader("Detected columns (index : name)")
    for idx, c in enumerate(cols_show):
        st.sidebar.write(f"{idx}: {repr(str(c))}")

    # Let user pick label column by index (or type name)
    label_col_index = st.sidebar.text_input("Label column index (leave blank if none). Use index shown above.", value="")
    label_col_option = None
    if label_col_index.strip() != "":
        try:
            idx = int(label_col_index)
            if idx < 0 or idx >= len(cols_show):
                st.sidebar.error("Invalid index")
                return
            label_col_option = cols_show[idx]
            st.sidebar.success(f"Using column index {idx} as label column -> {label_col_option!r}")
        except Exception:
            st.sidebar.error("Enter a valid integer index for label column")
            return
    else:
        label_col_option_text = st.sidebar.text_input("Or type label column name (if present)", value="label")
        if label_col_option_text.strip() != "":
            label_col_option = label_col_option_text.strip()

    # Preprocess the data (utils.preprocess_dataframe handles optional label)
    try:
        X, y = preprocess_dataframe(df, label_col=label_col_option if label_col_option else None)
    except Exception as e:
        st.error(f'Preprocessing failed: {e}')
        return

    # Supervised mode: require labels and run model predictions
    if mode.startswith("Supervised"):
        if y is None:
            st.error("No label column found. Upload a labeled CSV or switch to Unsupervised mode.")
            return

        # Align X to model feature names if present
        if feature_names is not None:
            missing = [f for f in feature_names if f not in X.columns]
            if missing:
                st.warning(f"Model expects features not present in upload: {missing}. Filling missing features with zeros.")
                for f in missing:
                    X[f] = 0
            X = X[feature_names]

        try:
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return

        df_results = df.copy()
        df_results['pred_label'] = preds
        if probs is not None:
            df_results['attack_prob'] = probs

    else:
        # Unsupervised: IsolationForest anomaly scoring
        st.info("Running unsupervised anomaly detection with IsolationForest (no labels required).")
        try:
            iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
            iso.fit(X)
            scores = iso.decision_function(X)  # higher -> more normal
            attack_score = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            df_results = df.copy()
            df_results['attack_score'] = attack_score
            df_results['pred_label'] = (attack_score > 0.5).astype(int)
        except Exception as e:
            st.error(f"IsolationForest failed: {e}")
            return

    # Show and allow download of results
    st.subheader('Predictions (first 100 rows)')
    st.dataframe(df_results.head(100))
    st.download_button('Download results CSV', df_results.to_csv(index=False).encode('utf-8'), 'nids_results.csv')

    # Summary
    st.subheader('Summary')
    try:
        st.write(df_results['pred_label'].value_counts().to_dict())
    except Exception:
        st.write("Could not compute summary counts.")

    # Feature importances (if supervised model supports them)
    if mode.startswith("Supervised") and hasattr(model, 'feature_importances_'):
        try:
            imp = model.feature_importances_
            f_names = feature_names if feature_names is not None else list(X.columns)
            imp_df = pd.DataFrame({'feature': f_names, 'importance': imp}).sort_values('importance', ascending=False).head(20)
            st.subheader('Top feature importances')
            st.bar_chart(imp_df.set_index('feature'))
        except Exception as e:
            st.info(f"Could not display feature importances: {e}")

    # Optional SHAP explanations (safe-guarded)
    if st.checkbox('Show SHAP explanations for first 5 rows (if supported)'):
        try:
            if mode.startswith("Supervised") and model is not None:
                explainer = shap.TreeExplainer(model)
                sample_X = X.iloc[:5]
                shap_values = explainer.shap_values(sample_X)
                st.write('SHAP values computed for first 5 rows')
                for i in range(min(5, sample_X.shape[0])):
                    st.write(f'Row {i}')
                    row_shap = shap_values[1][i] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values[i]
                    row_df = pd.DataFrame({'feature': sample_X.columns, 'shap_value': row_shap})
                    st.dataframe(row_df.sort_values('shap_value', key=abs, ascending=False).head(10))
            else:
                st.info("SHAP explanations are only available in Supervised mode with a compatible model.")
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")


if __name__ == '__main__':
    main()
