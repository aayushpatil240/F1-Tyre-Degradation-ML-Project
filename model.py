import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data_loader import load_race_laps


def train_regression_models():
    """
    Trains regression models to predict tyre degradation
    (lap_time_delta) based on lap_in_stint and compound.
    """

    # Load processed data
    df = load_race_laps()

    # Features & target
    X = df[["lap_in_stint", "Compound"]]
    y = df["lap_time_delta"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Preprocessing: one-hot encode compound
    preprocessor = ColumnTransformer(
        transformers=[
            ("compound", OneHotEncoder(drop="first"), ["Compound"])
        ],
        remainder="passthrough"
    )

    # -----------------------------
    # Model 1: Linear Regression
    # -----------------------------
    linear_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)

    # -----------------------------
    # Model 2: Random Forest
    # -----------------------------
    rf_model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    print("\n===== REGRESSION RESULTS =====\n")

    print("Linear Regression:")
    print("MAE:", round(mean_absolute_error(y_test, y_pred_lr), 3))
    print("R² :", round(r2_score(y_test, y_pred_lr), 3))

    print("\nRandom Forest:")
    print("MAE:", round(mean_absolute_error(y_test, y_pred_rf), 3))
    print("R² :", round(r2_score(y_test, y_pred_rf), 3))

    # -----------------------------
    # Visualization
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], linestyle="--")
    plt.xlabel("Actual Lap Time Delta (s)")
    plt.ylabel("Predicted Lap Time Delta (s)")
    plt.title("Tyre Degradation Prediction (Random Forest)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_regression_models()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_stints():
    """
    Clusters tyre stints based on degradation behaviour.
    """

    # Load processed lap-level data
    df = load_race_laps()

    # -----------------------------
    # Aggregate to stint-level
    # -----------------------------
    stint_df = (
        df.groupby(["stint_id", "Driver", "Compound"])
        .agg(
            stint_length=("lap_in_stint", "max"),
            avg_degradation_rate=("lap_time_delta", "mean"),
            total_pace_loss=("lap_time_delta", "max"),
        )
        .reset_index()
    )

    # Features for clustering
    X = stint_df[[
        "stint_length",
        "avg_degradation_rate",
        "total_pace_loss"
    ]]

    # Scale features (VERY important for KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # KMeans Clustering
    # -----------------------------
    kmeans = KMeans(n_clusters=3, random_state=42)
    stint_df["cluster"] = kmeans.fit_predict(X_scaled)

    # -----------------------------
    # Cluster interpretation
    # -----------------------------
    print("\n===== STINT CLUSTER SUMMARY =====\n")
    cluster_summary = (
        stint_df
        .groupby("cluster")[[
            "stint_length",
            "avg_degradation_rate",
            "total_pace_loss"
        ]]
        .mean()
        .round(3)
    )

    print(cluster_summary)

    print("\n===== SAMPLE STINTS PER CLUSTER =====\n")
    for c in sorted(stint_df["cluster"].unique()):
        print(f"\nCluster {c}:")
        print(
            stint_df[stint_df["cluster"] == c]
            .head(3)[
                ["stint_id", "Driver", "Compound", "stint_length", "total_pace_loss"]
            ]
        )


if __name__ == "__main__":
    # Comment ONE of these at a time if needed
    train_regression_models()
    cluster_stints()

