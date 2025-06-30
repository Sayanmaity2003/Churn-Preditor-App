import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
from datetime import datetime

# ---- Model download logic from Google Drive----
MODEL_PATH = "churn_model.pkl"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading churn_model.pkl from Google Drive..."):
        file_id = "12VYsvMMbCUn2SbtOiMDEIQGVg9-UJimT"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained model
model = joblib.load(MODEL_PATH)


st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Churn Prediction App")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# --- Preprocessing the data ---
def preprocess_input(features):
    features = features.copy()

    # Identify the date columns
    date_cols = [col for col in features.columns if "date" in col.lower()]
    for col in date_cols:
        features[col] = pd.to_datetime(features[col], errors='coerce')
        features[col + "_year"] = features[col].dt.year
        features[col + "_month"] = features[col].dt.month
        features[col + "_day"] = features[col].dt.day

    # Drop original date columns
    features.drop(columns=date_cols, inplace=True, errors="ignore")

    # Fill missing values
    features.fillna(0, inplace=True)

    # Encode categorical columns
    cat_cols = features.select_dtypes(include="object").columns
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True)

    # Drop any remaining datetime columns
    features = features.select_dtypes(exclude=["datetime64[ns]", "datetime64[ns, UTC]"])

    return features

# --- Prediction Logic ---
if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(data, use_container_width=True, height=600)

        # Preprocess input data
        X_input = preprocess_input(data)

        # Aligning columns with the model's expectations
        missing_cols = set(model.feature_names_in_) - set(X_input.columns)
        for col in missing_cols:
            X_input[col] = 0
        X_input = X_input[model.feature_names_in_]

        # Make predictions
        predictions = model.predict(X_input)
        data["Churn_Prediction"] = predictions

        st.success("Prediction Completed Successfully!")
        st.subheader("Churn Prediction Result")
        st.dataframe(data, use_container_width=True, height=600)

        # Download output
        csv_download = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction CSV", csv_download, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error during prediction: {e}!!")
