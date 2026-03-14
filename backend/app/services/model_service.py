from pathlib import Path
import joblib

MODEL_DIR = Path("app/models")

NB_PATH = MODEL_DIR / "naive_bayes.pkl"
DT_PATH = MODEL_DIR / "decision_tree.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"

_nb_model = None
_dt_model = None
_rf_model = None


def _load_models():
    global _nb_model, _dt_model, _rf_model

    if NB_PATH.exists():
        _nb_model = joblib.load(NB_PATH)
    if DT_PATH.exists():
        _dt_model = joblib.load(DT_PATH)
    if RF_PATH.exists():
        _rf_model = joblib.load(RF_PATH)


_load_models()


def _feature_vector(features: dict) -> list[float]:
    return [
        float(features["amount"]),
        float(features["hour_of_day"]),
        float(features["day_of_week"]),
        float(features["category_frequency"]),
        float(features["hour_frequency"]),
        float(features["distance_to_nearest_zone"]),
        float(features["in_frequent_zone"]),
        float(features["distance_to_phone"]),
        float(features["phone_near_transaction"]),
        float(features["category_is_rare"]),
        float(features["hour_is_rare"]),
        float(features["location_is_rare"]),
    ]


def _mock_score(features: dict) -> float:
    score = 0.15

    if features["amount"] > 200:
        score += 0.15
    if features["category_is_rare"]:
        score += 0.20
    if features["hour_is_rare"]:
        score += 0.10
    if features["location_is_rare"]:
        score += 0.20
    if features["distance_to_phone"] > 2:
        score += 0.20

    return min(score, 0.99)


def predict_model_scores(features: dict) -> dict:
    vector = [_feature_vector(features)]

    if _nb_model and _dt_model and _rf_model:
        nb_score = float(_nb_model.predict_proba(vector)[0][1])
        dt_score = float(_dt_model.predict_proba(vector)[0][1])
        rf_score = float(_rf_model.predict_proba(vector)[0][1])
    else:
        base = _mock_score(features)
        nb_score = max(0.0, min(base - 0.05, 1.0))
        dt_score = max(0.0, min(base, 1.0))
        rf_score = max(0.0, min(base + 0.05, 1.0))

    return {
        "naive_bayes": nb_score,
        "decision_tree": dt_score,
        "random_forest": rf_score,
    }