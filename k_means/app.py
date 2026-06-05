import streamlit as st
import numpy as np
import joblib

model = joblib.load(
    "models/kmeans_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊"
)

st.title(
    "📊 Customer Segmentation using K-Means"
)

st.write(
    "Predict customer cluster."
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
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

if st.button(
    "Predict Cluster"
):

    data = np.array(
        [
            [age, income, score]
        ]
    )

    data_scaled = scaler.transform(
        data
    )

    cluster = model.predict(
        data_scaled
    )[0]

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )

    if cluster == 0:
        st.info(
            "Average Customers"
        )

    elif cluster == 1:
        st.info(
            "High Spending Customers"
        )

    elif cluster == 2:
        st.info(
            "Premium Customers"
        )

    elif cluster == 3:
        st.info(
            "Budget Customers"
        )

    else:
        st.info(
            "Young Customers"
        )