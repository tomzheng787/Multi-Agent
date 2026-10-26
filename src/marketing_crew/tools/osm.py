"""OpenStreetMap (Overpass) nearby-business search used by the local search agent."""

from __future__ import annotations

import math
from typing import Any, Callable

from marketing_crew.tools.api_registry import ToolInputError

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

CATEGORY_TAGS: dict[str, list[tuple[str, str]]] = {
    "bakery": [("shop", "bakery")],
    "bar": [("amenity", "bar"), ("amenity", "pub")],
    "bookstore": [("shop", "books")],
    "cafe": [("amenity", "cafe")],
    "clothing": [("shop", "clothes")],
    "coworking": [("amenity", "coworking_space"), ("office", "coworking")],
    "fast_food": [("amenity", "fast_food")],
    "gym": [("leisure", "fitness_centre")],
    "hotel": [("tourism", "hotel")],
    "pet_store": [("shop", "pet")],
    "pharmacy": [("amenity", "pharmacy")],
    "restaurant": [("amenity", "restaurant")],
    "salon": [("shop", "hairdresser"), ("shop", "beauty")],
    "supermarket": [("shop", "supermarket"), ("shop", "convenience")],
    "yoga": [("sport", "yoga")],
}

MIN_RADIUS_M, MAX_RADIUS_M = 100, 5000


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def find_places(
    fetch: Callable[..., Any],
    *,
    lat: float,
    lon: float,
    category: str,
    radius_m: int = 1000,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Named businesses of ``category`` within ``radius_m`` of a point, nearest first."""
    if category not in CATEGORY_TAGS:
        raise ToolInputError(
            f"unknown category '{category}'",
            hint=f"Use one of: {', '.join(sorted(CATEGORY_TAGS))}.",
        )
    if not (MIN_RADIUS_M <= radius_m <= MAX_RADIUS_M):
        raise ToolInputError(
            f"radius_m={radius_m} is out of range",
            hint=f"radius_m must be between {MIN_RADIUS_M} and {MAX_RADIUS_M} metres.",
        )
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ToolInputError("lat/lon out of range", hint="Geocode the place first and pass its lat/lon.")

    selectors = "".join(f'nwr["{k}"="{v}"](around:{radius_m},{lat},{lon});' for k, v in CATEGORY_TAGS[category])
    query = f"[out:json][timeout:25];({selectors});out center {max(limit * 2, limit)};"
    payload = fetch("POST", OVERPASS_URL, data={"data": query})

    places = []
    for element in (payload or {}).get("elements", []):
        tags = element.get("tags") or {}
        name = tags.get("name")
        if not name:
            continue
        plat = element.get("lat", (element.get("center") or {}).get("lat"))
        plon = element.get("lon", (element.get("center") or {}).get("lon"))
        address = " ".join(
            part for part in (tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")) if part
        )
        places.append(
            {
                "name": name,
                "category": category,
                "address": address or None,
                "lat": plat,
                "lon": plon,
                "distance_m": round(_distance_m(lat, lon, plat, plon)) if plat is not None else None,
                "website": tags.get("website"),
                "opening_hours": tags.get("opening_hours"),
                "osm_id": f"{element.get('type')}/{element.get('id')}",
            }
        )
    places.sort(key=lambda p: p["distance_m"] if p["distance_m"] is not None else math.inf)
    return places[:limit]
