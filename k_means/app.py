import os
import joblib
import pandas as pd

if (
    not os.path.exists("models/kmeans_model.pkl")
    or
    not os.path.exists("models/scaler.pkl")
):

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(
        "data/Mall_Customers.csv"
    )

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

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/kmeans_model.pkl"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

model = joblib.load(
    "models/kmeans_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)
