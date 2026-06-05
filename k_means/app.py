import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# -----------------------------
# Train Model If Not Exists
# -----------------------------
if not MODEL_PATH.exists() or not SCALER_PATH.exists():

    df = pd.read_csv(DATA_PATH)

    X = df[
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Segmentation Using K-Means")

st.write(
    "Predict the customer cluster based on Age, Income and Spending Score."
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=1,
    max_value=200,
    value=50
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Cluster"):

    sample = np.array(
        [
            [age, income, score]
        ]
    )

    sample_scaled = scaler.transform(sample)

    cluster = model.predict(sample_scaled)[0]

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )

    if cluster == 0:
        st.info("Average Customers")

    elif cluster == 1:
        st.info("Premium Customers")

    elif cluster == 2:
        st.info("High Spending Customers")

    elif cluster == 3:
        st.info("Budget Customers")

    else:
        st.info("Young Customers")

# -----------------------------
# Dataset Preview
# -----------------------------
if st.checkbox("Show Dataset"):

    df = pd.read_csv(DATA_PATH)

    st.dataframe(df.head())
