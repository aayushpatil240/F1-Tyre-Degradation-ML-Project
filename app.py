import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_race_laps

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Tyre Degradation Analysis", layout="wide")

st.title("🏎️ Tyre Degradation & Strategy Analysis")
st.caption("ML-based race strategy insight using F1 race data")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = load_race_laps()

# Sidebar controls
st.sidebar.header("Race Filters")

driver = st.sidebar.selectbox(
    "Select Driver", sorted(df["Driver"].unique())
)

compound = st.sidebar.selectbox(
    "Select Compound", sorted(df["Compound"].unique())
)

# Filter data
filtered_df = df[
    (df["Driver"] == driver) &
    (df["Compound"] == compound)
]

if filtered_df.empty:
    st.warning("No data available for this selection.")
    st.stop()

# --------------------------------------------------
# Train regression model (same as Phase 3, lightweight)
# --------------------------------------------------
X = df[["lap_in_stint", "Compound"]]
y = df["lap_time_delta"]

preprocessor = ColumnTransformer(
    transformers=[
        ("compound", OneHotEncoder(drop="first"), ["Compound"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

model.fit(X, y)

# Predict for selected stint
filtered_df = filtered_df.copy()
filtered_df["predicted_delta"] = model.predict(
    filtered_df[["lap_in_stint", "Compound"]]
)

# --------------------------------------------------
# Cluster stints (same logic as Phase 4)
# --------------------------------------------------
stint_df = (
    df.groupby(["stint_id", "Driver", "Compound"])
    .agg(
        stint_length=("lap_in_stint", "max"),
        avg_degradation_rate=("lap_time_delta", "mean"),
        total_pace_loss=("lap_time_delta", "max"),
    )
    .reset_index()
)

scaler = StandardScaler()
X_cluster = scaler.fit_transform(
    stint_df[["stint_length", "avg_degradation_rate", "total_pace_loss"]]
)

kmeans = KMeans(n_clusters=3, random_state=42)
stint_df["cluster"] = kmeans.fit_predict(X_cluster)

# Map clusters to labels
cluster_means = stint_df.groupby("cluster")["total_pace_loss"].mean()
cluster_order = cluster_means.sort_values().index.tolist()

cluster_labels = {
    cluster_order[0]: "🟢 Conservative",
    cluster_order[1]: "🟡 Balanced",
    cluster_order[2]: "🔴 Aggressive",
}

stint_df["cluster_label"] = stint_df["cluster"].map(cluster_labels)

def generate_race_engineer_notes(stint_row, compound_avg):
    notes = []

    stint_length = stint_row["stint_length"]
    total_loss = stint_row["total_pace_loss"]
    avg_deg = stint_row["avg_degradation_rate"]
    cluster = stint_row["cluster_label"]

    # --- Overview ---
    notes.append(
        f"**Stint Overview:** {stint_length}-lap stint on {stint_row['Compound']} tyres "
        f"classified as **{cluster}**."
    )

    # --- Tyre management ---
    if total_loss > compound_avg["total_pace_loss"]:
        notes.append(
            "Tyre degradation was **higher than the average** for this compound, "
            "indicating increased thermal or mechanical stress."
        )
    else:
        notes.append(
            "Tyre degradation remained **under control** relative to the compound average."
        )

    # --- Pace behaviour ---
    if avg_deg > compound_avg["avg_degradation_rate"]:
        notes.append(
            "Pace drop-off per lap was **steeper than expected**, suggesting the tyre peak phase "
            "was followed by accelerated wear."
        )
    else:
        notes.append(
            "Pace degradation per lap was **stable**, indicating effective tyre usage through the stint."
        )

    # --- Strengths ---
    if stint_length >= compound_avg["stint_length"]:
        notes.append(
            "Driver successfully extended the stint length, extracting additional tyre life."
        )
    else:
        notes.append(
            "Stint length was shorter than optimal, limiting strategic flexibility."
        )

    # --- Weaknesses ---
    if "Aggressive" in cluster:
        notes.append(
            "Excessive tyre degradation limited the effective performance window of the stint."
        )

    # --- Recommendations ---
    recs = []
    if total_loss > compound_avg["total_pace_loss"]:
        recs.append("Reduce push laps during mid-stint phase")
    if avg_deg > compound_avg["avg_degradation_rate"]:
        recs.append("Introduce lift-and-coast or earlier tyre management phase")
    if not recs:
        recs.append("Maintain current tyre management approach")

    notes.append("**Recommendations:** " + "; ".join(recs) + ".")

    return notes


current_stint = filtered_df["stint_id"].iloc[0]
cluster_label = stint_df[
    stint_df["stint_id"] == current_stint
]["cluster_label"].values[0]

# --------------------------------------------------
# Plots
# --------------------------------------------------
st.subheader(f"Driver: {driver} | Compound: {compound}")
st.markdown(f"**Stint Strategy:** {cluster_label}")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    filtered_df["lap_in_stint"],
    filtered_df["lap_time_delta"],
    label="Actual Degradation",
    marker="o"
)

ax.plot(
    filtered_df["lap_in_stint"],
    filtered_df["predicted_delta"],
    label="Predicted Degradation",
    linestyle="--"
)

ax.set_xlabel("Lap in Stint")
ax.set_ylabel("Lap Time Delta (s)")
ax.set_title("Tyre Degradation Curve")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Comparison view
# --------------------------------------------------
st.subheader("📊 Comparison with Other Stints (Same Compound)")

compare_df = stint_df[stint_df["Compound"] == compound]

st.dataframe(
    compare_df[[
        "stint_id",
        "Driver",
        "stint_length",
        "total_pace_loss",
        "cluster_label"
    ]].sort_values("total_pace_loss")
)

st.subheader("📝 Race Engineer Notes")

# Current stint row
current_stint_row = stint_df.loc[
    stint_df["stint_id"] == current_stint
].squeeze()


# Compound averages
compound_avg = (
    stint_df[stint_df["Compound"] == compound]
    [["stint_length", "avg_degradation_rate", "total_pace_loss"]]
    .mean()
)

notes = generate_race_engineer_notes(current_stint_row, compound_avg)

for note in notes:
    st.markdown(f"- {note}")
