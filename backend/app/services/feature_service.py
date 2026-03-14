from datetime import datetime
from app.utils.geo import haversine_km


def _get_hour_and_day(timestamp: str) -> tuple[int, int]:
    dt = datetime.fromisoformat(timestamp)
    return dt.hour, dt.weekday()


def _get_category_frequency(profile: dict, merchant_category: str) -> float:
    return profile.get("common_categories", {}).get(merchant_category.lower(), 0.0)


def _get_hour_frequency(profile: dict, hour: int) -> float:
    return profile.get("common_hours", {}).get(str(hour), 0.0)


def _get_distance_to_nearest_zone(profile: dict, tx_lat: float, tx_lng: float) -> float:
    frequent_zones = profile.get("frequent_zones", [])
    if not frequent_zones:
        return 9999.0

    min_distance = float("inf")
    for zone in frequent_zones:
        distance = haversine_km(tx_lat, tx_lng, zone["lat"], zone["lng"])
        min_distance = min(min_distance, distance)

    return min_distance


def _is_in_any_zone(profile: dict, tx_lat: float, tx_lng: float) -> bool:
    frequent_zones = profile.get("frequent_zones", [])
    for zone in frequent_zones:
        distance = haversine_km(tx_lat, tx_lng, zone["lat"], zone["lng"])
        if distance <= zone["radius_km"]:
            return True
    return False


def build_features(transaction, profile: dict) -> dict:
    hour_of_day, day_of_week = _get_hour_and_day(transaction.timestamp)
    category_frequency = _get_category_frequency(profile, transaction.merchant_category)
    hour_frequency = _get_hour_frequency(profile, hour_of_day)
    distance_to_nearest_zone = _get_distance_to_nearest_zone(
        profile,
        transaction.transaction_lat,
        transaction.transaction_lng
    )
    in_frequent_zone = _is_in_any_zone(
        profile,
        transaction.transaction_lat,
        transaction.transaction_lng
    )
    distance_to_phone = haversine_km(
        transaction.transaction_lat,
        transaction.transaction_lng,
        transaction.phone_lat,
        transaction.phone_lng
    )

    phone_near_transaction = distance_to_phone <= 0.5
    category_is_rare = category_frequency < 0.05
    hour_is_rare = hour_frequency < 0.03
    location_is_rare = distance_to_nearest_zone > 5.0

    return {
        "amount": transaction.amount,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "merchant_category": transaction.merchant_category.lower(),
        "category_frequency": category_frequency,
        "hour_frequency": hour_frequency,
        "distance_to_nearest_zone": distance_to_nearest_zone,
        "in_frequent_zone": int(in_frequent_zone),
        "distance_to_phone": distance_to_phone,
        "phone_near_transaction": int(phone_near_transaction),
        "category_is_rare": int(category_is_rare),
        "hour_is_rare": int(hour_is_rare),
        "location_is_rare": int(location_is_rare),
    }