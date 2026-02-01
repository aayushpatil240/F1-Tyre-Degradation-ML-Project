import fastf1
import pandas as pd
import os

# -----------------------------
# Cache setup
# -----------------------------
CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_laps(season=2024, race_name="Bahrain Grand Prix"):
    """
    Loads race lap data and performs basic feature engineering.
    Returns lap-level dataframe with stint features.
    """

    # Load race session
    session = fastf1.get_session(season, race_name, "R")
    session.load()

    laps = session.laps

    # Keep only relevant columns
    laps = laps[[
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "Stint"
    ]]

    # Drop invalid rows
    laps = laps.dropna(subset=["LapTime", "Compound"])

    # Convert lap time to seconds
    laps["lap_time"] = laps["LapTime"].dt.total_seconds()

    # Sort laps correctly
    laps = laps.sort_values(by=["Driver", "Stint", "LapNumber"]).reset_index(drop=True)

    # Create stint_id
    laps["stint_id"] = laps["Driver"] + "_" + laps["Stint"].astype(int).astype(str)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # Lap number within stint
    laps["lap_in_stint"] = laps.groupby("stint_id").cumcount() + 1

    # Best (fastest) lap time in each stint
    laps["stint_best_time"] = laps.groupby("stint_id")["lap_time"].transform("min")

# Lap time delta vs best lap (true degradation)
    laps["lap_time_delta"] = laps["lap_time"] - laps["stint_best_time"]

# Drop helper columns
    laps = laps.drop(columns=["LapTime", "stint_best_time"])


    return laps


if __name__ == "__main__":
    df = load_race_laps()

    print("\n===== PHASE 2: FEATURE ENGINEERING COMPLETE =====\n")
    print("Total laps:", len(df))
    print("Total stints:", df["stint_id"].nunique())

    print("\n===== SAMPLE DATA =====\n")
    print(df.head(12))

    print("\n===== LAP-IN-STINT CHECK =====\n")
    print(df.groupby("stint_id")["lap_in_stint"].max().head())

    print("\n===== LAP TIME DELTA STATS =====\n")
    print(df["lap_time_delta"].describe())
