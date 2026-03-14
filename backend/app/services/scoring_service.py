def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(value, max_value))


def compute_ml_score(model_scores: dict) -> float:
    return (
        0.2 * model_scores["naive_bayes"]
        + 0.3 * model_scores["decision_tree"]
        + 0.5 * model_scores["random_forest"]
    )


def compute_behavior_penalty(features: dict) -> float:
    penalty = 0.0

    if features["category_frequency"] < 0.05:
        penalty += 0.12

    if features["hour_frequency"] < 0.03:
        penalty += 0.08

    return penalty


def compute_geo_penalty(features: dict) -> float:
    distance = features["distance_to_nearest_zone"]

    if distance > 10:
        return 0.15
    if distance > 5:
        return 0.08
    return 0.0


def compute_phone_match_reduction(features: dict) -> float:
    distance = features["distance_to_phone"]

    if distance < 0.5:
        return 0.18
    if distance < 2.0:
        return 0.08
    return 0.0


def build_reasons(features: dict) -> list[str]:
    reasons = []

    if features["category_frequency"] < 0.05:
        reasons.append("merchant category uncommon for this user")

    if features["hour_frequency"] < 0.03:
        reasons.append("transaction time unusual for this user")

    if features["distance_to_nearest_zone"] > 5:
        reasons.append("transaction outside frequent spending zone")

    if features["distance_to_phone"] > 2:
        reasons.append("phone location not close to purchase")

    if not reasons:
        reasons.append("transaction matches normal user behavior")

    return reasons


def get_label(final_score: float) -> str:
    if final_score >= 0.75:
        return "HIGH_RISK"
    if final_score >= 0.45:
        return "MEDIUM_RISK"
    return "LOW_RISK"


def compute_final_decision(features: dict, model_scores: dict) -> dict:
    ml_score = compute_ml_score(model_scores)
    behavior_penalty = compute_behavior_penalty(features)
    geo_penalty = compute_geo_penalty(features)
    phone_match_reduction = compute_phone_match_reduction(features)

    final_score = ml_score + behavior_penalty + geo_penalty - phone_match_reduction
    final_score = clamp(final_score)

    reasons = build_reasons(features)
    label = get_label(final_score)
    #TODO: Add more heatmap information for a LLM to ingest alongisde input data
    return {
        "ml_score": round(ml_score, 4),
        "behavior_penalty": round(behavior_penalty, 4),
        "geo_penalty": round(geo_penalty, 4),
        "phone_match_reduction": round(phone_match_reduction, 4),
        "heatmap_score": round(behavior_penalty + geo_penalty, 4),
        "final_score": round(final_score, 4),
        "label": label,
        "reasons": reasons,
    }