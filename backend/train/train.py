from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "category_frequency",
    "hour_frequency",
    "distance_to_nearest_zone",
    "in_frequent_zone",
    "distance_to_phone",
    "phone_near_transaction",
    "category_is_rare",
    "hour_is_rare",
    "location_is_rare",
]

TARGET_COLUMN = "is_fraud"

REQUIRED_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "train" / "credit_card_transactions.csv"
ENGINEERED_OUTPUT = BASE_DIR / "train" / "engineered_transactions.csv"
PROFILE_OUTPUT = BASE_DIR / "data" / "user_profiles.json"

MODEL_DIR = BASE_DIR / "app" / "models"
NB_PATH = MODEL_DIR / "naive_bayes.pkl"
DT_PATH = MODEL_DIR / "decision_tree.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate engineered fraud features and train models."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to fraudTrain.csv",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=8,
        help="How many users (cc_num values) to select for MVP training",
    )
    parser.add_argument(
        "--max-transactions-per-user",
        type=int,
        default=400,
        help="Max number of legitimate transactions kept per selected user",
    )
    parser.add_argument(
        "--synthetic-fraud-ratio",
        type=float,
        default=0.35,
        help="How many synthetic fraud rows to create relative to legit rows",
    )
    parser.add_argument(
        "--clusters-per-user",
        type=int,
        default=2,
        help="Number of merchant-location clusters per user",
    )
    parser.add_argument(
        "--save-engineered-csv",
        action="store_true",
        help="Whether to save engineered dataset CSV",
    )
    return parser.parse_args()


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df = df.dropna(subset=["trans_date_trans_time", "cc_num", "category", "amt", "merch_lat", "merch_long"])

    df["cc_num"] = df["cc_num"].astype(str)
    df["category"] = df["category"].astype(str).str.lower()
    df["merchant"] = df["merchant"].astype(str)
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["merch_lat"] = pd.to_numeric(df["merch_lat"], errors="coerce")
    df["merch_long"] = pd.to_numeric(df["merch_long"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["amt", "lat", "long", "merch_lat", "merch_long"])

    df["hour_of_day"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    return df


def choose_users(df: pd.DataFrame, num_users: int, max_transactions_per_user: int) -> pd.DataFrame:
    legit = df[df["is_fraud"] == 0].copy()

    counts = (
        legit.groupby("cc_num")
        .size()
        .reset_index(name="txn_count")
        .sort_values("txn_count", ascending=False)
    )

    eligible = counts[counts["txn_count"] >= max(50, min(100, max_transactions_per_user // 2))]
    selected_users = eligible["cc_num"].head(num_users).tolist()

    if not selected_users:
        raise ValueError("No eligible users found. Reduce --num-users or --max-transactions-per-user.")

    subset = legit[legit["cc_num"].isin(selected_users)].copy()
    subset = subset.sort_values(["cc_num", "trans_date_trans_time"])

    sampled_parts = []
    for cc_num, group in subset.groupby("cc_num"):
        if len(group) > max_transactions_per_user:
            sampled = group.sample(max_transactions_per_user, random_state=RANDOM_SEED)
            sampled = sampled.sort_values("trans_date_trans_time")
        else:
            sampled = group
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts, ignore_index=True)
    return result


def build_user_profile(user_df: pd.DataFrame, clusters_per_user: int) -> dict[str, Any]:
    category_freq = user_df["category"].value_counts(normalize=True).to_dict()
    hour_freq = user_df["hour_of_day"].value_counts(normalize=True).sort_index().to_dict()

    coords = user_df[["merch_lat", "merch_long"]].to_numpy()

    n_clusters = min(clusters_per_user, len(coords))
    if n_clusters <= 0:
        zone_centers = []
        zone_radii = []
    elif n_clusters == 1:
        center = coords.mean(axis=0)
        dists = [haversine_km(center[0], center[1], row[0], row[1]) for row in coords]
        radius = float(np.percentile(dists, 80)) if dists else 2.0
        zone_centers = [center]
        zone_radii = [max(radius, 1.0)]
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_

        zone_centers = []
        zone_radii = []

        for cluster_idx in range(n_clusters):
            cluster_points = coords[labels == cluster_idx]
            center = centers[cluster_idx]
            dists = [
                haversine_km(center[0], center[1], point[0], point[1])
                for point in cluster_points
            ]
            radius = float(np.percentile(dists, 80)) if len(dists) > 0 else 2.0
            zone_centers.append(center)
            zone_radii.append(max(radius, 1.0))

    frequent_zones = []
    for i, center in enumerate(zone_centers):
        frequent_zones.append(
            {
                "name": f"zone_{i+1}",
                "lat": float(center[0]),
                "lng": float(center[1]),
                "radius_km": float(zone_radii[i]),
            }
        )

    return {
        "common_categories": {str(k): float(v) for k, v in category_freq.items()},
        "common_hours": {str(int(k)): float(v) for k, v in hour_freq.items()},
        "frequent_zones": frequent_zones,
    }


def compute_distance_to_nearest_zone(lat: float, lng: float, frequent_zones: list[dict[str, Any]]) -> float:
    if not frequent_zones:
        return 9999.0

    distances = [
        haversine_km(lat, lng, zone["lat"], zone["lng"])
        for zone in frequent_zones
    ]
    return float(min(distances))


def compute_in_frequent_zone(lat: float, lng: float, frequent_zones: list[dict[str, Any]]) -> int:
    for zone in frequent_zones:
        distance = haversine_km(lat, lng, zone["lat"], zone["lng"])
        if distance <= float(zone["radius_km"]):
            return 1
    return 0


def simulate_phone_location(
    tx_lat: float,
    tx_lng: float,
    is_fraud: int,
) -> tuple[float, float]:
    if is_fraud == 0:
        # Close to transaction
        phone_lat = tx_lat + np.random.normal(0, 0.08)
        phone_lng = tx_lng + np.random.normal(0, 0.08)
    else:
        # Deliberately farther away
        phone_lat = tx_lat + np.random.normal(4.0, 1.5)
        phone_lng = tx_lng + np.random.normal(4.0, 1.5)

    phone_lat = clamp(phone_lat, -89.0, 89.0)
    phone_lng = clamp(phone_lng, -179.0, 179.0)
    return float(phone_lat), float(phone_lng)


def pick_rare_category(all_categories: list[str], common_categories: dict[str, float]) -> str:
    rare_candidates = [cat for cat in all_categories if common_categories.get(cat, 0.0) < 0.02]
    if rare_candidates:
        return random.choice(rare_candidates)
    return random.choice(all_categories)


def mutate_to_synthetic_fraud(
    legit_row: pd.Series,
    user_profile: dict[str, Any],
    all_categories: list[str],
) -> dict[str, Any]:
    tx_time = legit_row["trans_date_trans_time"]

    fraud_type = random.choice(["geo", "behavior", "combined"])

    mutated = legit_row.to_dict()
    mutated["is_fraud"] = 1

    # Default starting point: merchant coords from original row
    merch_lat = float(mutated["merch_lat"])
    merch_lng = float(mutated["merch_long"])
    amount = float(mutated["amt"])
    category = str(mutated["category"]).lower()

    # Make time unusual
    if fraud_type in ("behavior", "combined"):
        suspicious_hour = random.choice([0, 1, 2, 3, 4, 23])
        mutated["trans_date_trans_time"] = tx_time.replace(hour=suspicious_hour)

    # Make category unusual
    if fraud_type in ("behavior", "combined"):
        category = pick_rare_category(all_categories, user_profile["common_categories"])
        mutated["category"] = category

    # Make amount more suspicious
    if fraud_type in ("behavior", "combined"):
        amount_multiplier = random.uniform(2.0, 5.0)
        mutated["amt"] = round(amount * amount_multiplier, 2)

    # Make location unusual
    if fraud_type in ("geo", "combined"):
        lat_shift = random.choice([-1, 1]) * random.uniform(3.0, 12.0)
        lng_shift = random.choice([-1, 1]) * random.uniform(3.0, 12.0)
        mutated["merch_lat"] = clamp(merch_lat + lat_shift, -89.0, 89.0)
        mutated["merch_long"] = clamp(merch_lng + lng_shift, -179.0, 179.0)

    # Refresh time-derived fields
    mutated_time = pd.to_datetime(mutated["trans_date_trans_time"])
    mutated["hour_of_day"] = int(mutated_time.hour)
    mutated["day_of_week"] = int(mutated_time.dayofweek)

    return mutated


def engineer_feature_row(row: pd.Series | dict[str, Any], user_profile: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    else:
        row_dict = row

    tx_lat = float(row_dict["merch_lat"])
    tx_lng = float(row_dict["merch_long"])
    hour_of_day = int(row_dict["hour_of_day"])
    day_of_week = int(row_dict["day_of_week"])
    category = str(row_dict["category"]).lower()

    category_frequency = float(user_profile["common_categories"].get(category, 0.0))
    hour_frequency = float(user_profile["common_hours"].get(str(hour_of_day), 0.0))

    distance_to_nearest_zone = compute_distance_to_nearest_zone(
        tx_lat,
        tx_lng,
        user_profile["frequent_zones"],
    )
    in_frequent_zone = compute_in_frequent_zone(
        tx_lat,
        tx_lng,
        user_profile["frequent_zones"],
    )

    phone_lat, phone_lng = simulate_phone_location(
        tx_lat,
        tx_lng,
        int(row_dict["is_fraud"]),
    )

    distance_to_phone = haversine_km(tx_lat, tx_lng, phone_lat, phone_lng)
    phone_near_transaction = int(distance_to_phone <= 0.5)

    category_is_rare = int(category_frequency < 0.05)
    hour_is_rare = int(hour_frequency < 0.03)
    location_is_rare = int(distance_to_nearest_zone > 5.0)

    return {
        "user_id": str(row_dict["cc_num"]),
        "transaction_id": str(row_dict.get("merchant", "txn")) + "_" + str(random.randint(100000, 999999)),
        "merchant_name": str(row_dict["merchant"]),
        "merchant_category": category,
        "transaction_lat": tx_lat,
        "transaction_lng": tx_lng,
        "phone_lat": float(phone_lat),
        "phone_lng": float(phone_lng),
        "timestamp": str(pd.to_datetime(row_dict["trans_date_trans_time"]).isoformat()),
        "amount": float(row_dict["amt"]),
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "category_frequency": category_frequency,
        "hour_frequency": hour_frequency,
        "distance_to_nearest_zone": float(distance_to_nearest_zone),
        "in_frequent_zone": int(in_frequent_zone),
        "distance_to_phone": float(distance_to_phone),
        "phone_near_transaction": int(phone_near_transaction),
        "category_is_rare": int(category_is_rare),
        "hour_is_rare": int(hour_is_rare),
        "location_is_rare": int(location_is_rare),
        "is_fraud": int(row_dict["is_fraud"]),
    }


def build_profiles_and_dataset(
    legit_df: pd.DataFrame,
    synthetic_fraud_ratio: float,
    clusters_per_user: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    all_categories = sorted(legit_df["category"].dropna().astype(str).str.lower().unique().tolist())

    profiles_for_api: list[dict[str, Any]] = []
    engineered_rows: list[dict[str, Any]] = []

    for cc_num, user_legit_df in legit_df.groupby("cc_num"):
        user_legit_df = user_legit_df.sort_values("trans_date_trans_time").copy()
        user_profile = build_user_profile(user_legit_df, clusters_per_user)

        profiles_for_api.append(
            {
                "user_id": str(cc_num),
                "common_categories": user_profile["common_categories"],
                "common_hours": user_profile["common_hours"],
                "frequent_zones": user_profile["frequent_zones"],
            }
        )

        # Legit rows
        for _, row in user_legit_df.iterrows():
            engineered_rows.append(engineer_feature_row(row, user_profile))

        # Synthetic fraud rows
        num_synthetic = max(1, int(len(user_legit_df) * synthetic_fraud_ratio))
        sampled_for_mutation = user_legit_df.sample(
            n=min(num_synthetic, len(user_legit_df)),
            random_state=RANDOM_SEED,
            replace=False,
        )

        for _, row in sampled_for_mutation.iterrows():
            fraud_row = mutate_to_synthetic_fraud(row, user_profile, all_categories)
            engineered_rows.append(engineer_feature_row(fraud_row, user_profile))

    engineered_df = pd.DataFrame(engineered_rows)
    return engineered_df, profiles_for_api


def train_models(engineered_df: pd.DataFrame) -> tuple[Any, Any, Any]:
    X = engineered_df[FEATURE_COLUMNS].copy()
    y = engineered_df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    nb_model = GaussianNB()
    dt_model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=8,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    nb_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    for name, model in [
        ("Naive Bayes", nb_model),
        ("Decision Tree", dt_model),
        ("Random Forest", rf_model),
    ]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=4))
        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC: {auc:.4f}")
        except ValueError:
            print("ROC-AUC unavailable.")

    return nb_model, dt_model, rf_model


def save_models(nb_model: Any, dt_model: Any, rf_model: Any) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(nb_model, NB_PATH)
    joblib.dump(dt_model, DT_PATH)
    joblib.dump(rf_model, RF_PATH)

    print("\nSaved model files:")
    print(f" - {NB_PATH}")
    print(f" - {DT_PATH}")
    print(f" - {RF_PATH}")


def save_profiles_for_api(profiles: list[dict[str, Any]]) -> None:
    PROFILE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nSaved user profiles for API: {PROFILE_OUTPUT}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)

    print(f"Loading dataset: {csv_path}")
    raw_df = load_raw_dataset(csv_path)
    print(f"Loaded rows: {len(raw_df):,}")

    legit_subset = choose_users(
        raw_df,
        num_users=args.num_users,
        max_transactions_per_user=args.max_transactions_per_user,
    )
    print(f"Selected legitimate rows for MVP: {len(legit_subset):,}")
    print(f"Selected users: {legit_subset['cc_num'].nunique()}")

    engineered_df, profiles = build_profiles_and_dataset(
        legit_subset,
        synthetic_fraud_ratio=args.synthetic_fraud_ratio,
        clusters_per_user=args.clusters_per_user,
    )

    print(f"Engineered rows: {len(engineered_df):,}")
    print(engineered_df[TARGET_COLUMN].value_counts(dropna=False).sort_index())

    # Save engineered CSV if requested
    if args.save_engineered_csv:
        ENGINEERED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        engineered_df.to_csv(ENGINEERED_OUTPUT, index=False)
        print(f"\nSaved engineered dataset: {ENGINEERED_OUTPUT}")

    save_profiles_for_api(profiles)

    nb_model, dt_model, rf_model = train_models(engineered_df)
    save_models(nb_model, dt_model, rf_model)


    print("Restart the FastAPI server")


if __name__ == "__main__":
    main()