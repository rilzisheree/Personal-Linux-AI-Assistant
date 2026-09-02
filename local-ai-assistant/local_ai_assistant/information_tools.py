"""Current-information services used by Lura's tool registry.

The module intentionally uses the Python standard library only. Each public
function returns a structured, JSON-encoded result so the model can summarize
it without being given permission to run arbitrary network or shell commands.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import socket
from dataclasses import dataclass
from datetime import date as date_type
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


USER_AGENT = "Lura/1.0 (personal assistant; current-information tools)"
REQUEST_TIMEOUT = 15
MAX_RESPONSE_BYTES = 1_500_000

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


@dataclass(frozen=True)
class InformationResult:
    success: bool
    content: str


class _SearchParser(HTMLParser):
    """Read result cards from DuckDuckGo's lightweight HTML endpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._field: str | None = None
        self._field_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current = {
                "title": "",
                "url": attributes.get("href") or "",
                "snippet": "",
            }
            self._field = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._field = "snippet"
            self._field_tag = tag

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._field:
            self._current[self._field] += " ".join(data.split())

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "a" and self._field == "title":
            self._field = None
        elif self._field == "snippet" and tag == self._field_tag:
            self.items.append(self._current)
            self._current = None
            self._field = None
            self._field_tag = None


def _request_bytes(url: str, service: str, accept: str) -> bytes:
    request = Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read(MAX_RESPONSE_BYTES)
    except HTTPError as error:
        raise ValueError(f"{service} returned HTTP {error.code}.") from error
    except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as error:
        raise ValueError(f"{service} is unavailable: {error}") from error


def _request_json(url: str, service: str) -> dict | list:
    raw = _request_bytes(url, service, "application/json")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{service} returned invalid JSON.") from error
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"{service} returned an invalid response.")
    return payload


def _request_xml(url: str, service: str) -> ElementTree.Element:
    raw = _request_bytes(url, service, "application/rss+xml, application/xml, text/xml")
    try:
        return ElementTree.fromstring(raw)
    except (ElementTree.ParseError, UnicodeDecodeError) as error:
        raise ValueError(f"{service} returned invalid XML.") from error


def _result(service: str, operation) -> InformationResult:
    try:
        payload = operation()
        return InformationResult(
            True,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return InformationResult(False, f"{service} failed: {error}")


def _clean_html(value: object) -> str:
    text = html.unescape(str(value or ""))
    return re.sub(r"<[^>]*>", "", text).strip()


def _first_text(element: ElementTree.Element, name: str) -> str:
    child = element.find(name)
    return (child.text or "").strip() if child is not None else ""


def _geocode(query: str, limit: int = 5) -> list[dict]:
    url = "https://nominatim.openstreetmap.org/search?" + urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "addressdetails": "1",
            "limit": limit,
        }
    )
    payload = _request_json(url, "OpenStreetMap geocoding")
    if not isinstance(payload, list):
        raise ValueError("OpenStreetMap returned an invalid place list.")
    places = [place for place in payload if isinstance(place, dict)]
    if not places:
        raise ValueError(f"No location matched '{query}'.")
    return places


def _location_label(place: dict) -> str:
    address = place.get("address")
    if isinstance(address, dict):
        parts = [
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality"),
            address.get("state"),
            address.get("country"),
        ]
        label = ", ".join(str(part) for part in parts if part)
        if label:
            return label
    return str(place.get("display_name") or "the requested location")


def _coordinates(place: dict) -> tuple[float, float]:
    try:
        return float(place["lon"]), float(place["lat"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The geocoding service returned invalid coordinates.") from error


def weather(location: str, requested_date: str, units: str) -> InformationResult:
    def operation() -> dict:
        query = location.strip() or ""
        if not query:
            query = os.environ.get("LURA_DEFAULT_LOCATION", "").strip()
        if not query:
            raise ValueError(
                "A city or location is required. Set LURA_DEFAULT_LOCATION to choose a default."
            )
        place = _geocode(query, 1)[0]
        longitude, latitude = _coordinates(place)
        normalized_units = units.strip().casefold() or "metric"
        if normalized_units not in {"metric", "imperial"}:
            raise ValueError("Units must be metric or imperial.")
        temperature_unit = "fahrenheit" if normalized_units == "imperial" else "celsius"
        windspeed_unit = "mph" if normalized_units == "imperial" else "kmh"
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(
                    (
                        "temperature_2m",
                        "apparent_temperature",
                        "relative_humidity_2m",
                        "precipitation",
                        "weather_code",
                        "wind_speed_10m",
                    )
                ),
                "daily": ",".join(
                    (
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "sunrise",
                        "sunset",
                    )
                ),
                "temperature_unit": temperature_unit,
                "wind_speed_unit": windspeed_unit,
                "timezone": "auto",
                "forecast_days": 16,
            }
        )
        payload = _request_json(url, "Open-Meteo weather")
        if not isinstance(payload, dict):
            raise ValueError("Open-Meteo returned an invalid forecast.")
        current = payload.get("current")
        daily = payload.get("daily")
        if not isinstance(current, dict) or not isinstance(daily, dict):
            raise ValueError("Open-Meteo returned incomplete forecast data.")
        dates = daily.get("time")
        if not isinstance(dates, list):
            raise ValueError("Open-Meteo returned no forecast dates.")
        normalized_date = requested_date.strip().casefold()
        if normalized_date in {"", "today", "now"}:
            target_date = dates[0]
        elif normalized_date == "tomorrow":
            target_date = dates[1] if len(dates) > 1 else dates[0]
        else:
            try:
                target_date = date_type.fromisoformat(normalized_date).isoformat()
            except ValueError as error:
                raise ValueError("Date must be today, tomorrow, or YYYY-MM-DD.") from error
        if target_date not in dates:
            raise ValueError(f"No forecast is available for {target_date}.")
        index = dates.index(target_date)

        def daily_value(name: str) -> object:
            values = daily.get(name)
            return values[index] if isinstance(values, list) and index < len(values) else None

        weather_code = daily_value("weather_code")
        current_code = current.get("weather_code")
        return {
            "location": _location_label(place),
            "requested_date": target_date,
            "timezone": payload.get("timezone"),
            "current": {
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed": current.get("wind_speed_10m"),
                "condition": WEATHER_CODES.get(current_code, "unknown"),
            },
            "forecast": {
                "condition": WEATHER_CODES.get(weather_code, "unknown"),
                "high": daily_value("temperature_2m_max"),
                "low": daily_value("temperature_2m_min"),
                "rain_probability_percent": daily_value("precipitation_probability_max"),
                "sunrise": daily_value("sunrise"),
                "sunset": daily_value("sunset"),
            },
            "units": normalized_units,
            "source": "Open-Meteo",
        }

    return _result("Weather", operation)


def news(query: str, max_results: int) -> InformationResult:
    def operation() -> dict:
        topic = query.strip() or "latest news"
        url = "https://news.google.com/rss/search?" + urlencode(
            {"q": topic, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        root = _request_xml(url, "Google News")
        items: list[dict[str, str]] = []
        for item in root.findall(".//item")[:max_results]:
            title = _first_text(item, "title")
            link = _first_text(item, "link")
            published = _first_text(item, "pubDate")
            description = _clean_html(_first_text(item, "description"))
            if title:
                items.append(
                    {
                        "title": title,
                        "published": published,
                        "summary": description,
                        "url": link,
                    }
                )
        if not items:
            raise ValueError(f"No recent news was found for '{topic}'.")
        return {"query": topic, "results": items, "source": "Google News RSS"}

    return _result("News search", operation)


def knowledge_search(query: str, max_results: int) -> InformationResult:
    def operation() -> dict:
        topic = query.strip()
        if not topic:
            raise ValueError("A knowledge-search query is required.")
        url = "https://en.wikipedia.org/w/api.php?" + urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srlimit": max_results,
                "format": "json",
                "utf8": "1",
                "origin": "*",
            }
        )
        payload = _request_json(url, "Wikipedia")
        search_data = payload.get("query") if isinstance(payload, dict) else None
        raw_results = search_data.get("search") if isinstance(search_data, dict) else None
        if not isinstance(raw_results, list) or not raw_results:
            raise ValueError(f"Wikipedia found no result for '{topic}'.")
        results: list[dict[str, str]] = []
        for raw in raw_results[:max_results]:
            if not isinstance(raw, dict) or not raw.get("title"):
                continue
            title = str(raw["title"])
            results.append(
                {
                    "title": title,
                    "snippet": _clean_html(raw.get("snippet")),
                    "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                }
            )
        if results:
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(results[0]["title"].replace(" ", "_"))
            )
            try:
                summary = _request_json(summary_url, "Wikipedia summary")
                if isinstance(summary, dict) and summary.get("extract"):
                    results[0]["summary"] = str(summary["extract"])
            except ValueError:
                # Search results remain useful if the optional summary endpoint
                # is rate-limited or unavailable.
                pass
        return {"query": topic, "results": results, "source": "Wikipedia"}

    return _result("Knowledge search", operation)


def currency(amount: float, from_currency: str, to_currency: str) -> InformationResult:
    def operation() -> dict:
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise ValueError("Amount must be a number.")
        if not math.isfinite(float(amount)):
            raise ValueError("Amount must be finite.")
        if float(amount) < 0:
            raise ValueError("Amount cannot be negative.")
        source = from_currency.strip().upper()
        target = to_currency.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", source) or not re.fullmatch(r"[A-Z]{3}", target):
            raise ValueError("Currencies must be three-letter ISO 4217 codes.")
        if source == target:
            return {
                "amount": float(amount),
                "from": source,
                "to": target,
                "converted_amount": float(amount),
                "rate": 1,
                "source": "same-currency calculation",
            }
        url = "https://open.er-api.com/v6/latest/" + quote(source)
        payload = _request_json(url, "ExchangeRate-API")
        if not isinstance(payload, dict):
            raise ValueError("ExchangeRate-API returned an invalid exchange-rate response.")
        rates = payload.get("rates")
        rate = rates.get(target) if isinstance(rates, dict) else None
        if payload.get("result") not in {None, "success"}:
            raise ValueError(str(payload.get("error-type") or "the provider rejected the request"))
        converted = float(amount) * float(rate) if isinstance(rate, (int, float)) else None
        if not isinstance(converted, (int, float)):
            raise ValueError(f"No current rate was returned for {source} to {target}.")
        return {
            "amount": float(amount),
            "from": source,
            "to": target,
            "converted_amount": converted,
            "rate": rate,
            "rate_date": payload.get("time_last_update_utc") or payload.get("date"),
            "source": "ExchangeRate-API",
        }

    return _result("Currency conversion", operation)


def find_places(query: str, near: str, max_results: int) -> InformationResult:
    def operation() -> dict:
        search_query = query.strip()
        if not search_query:
            raise ValueError("A place or business query is required.")
        if near.strip():
            search_query = f"{search_query}, {near.strip()}"
        places = _geocode(search_query, max_results)
        results = []
        for place in places:
            longitude, latitude = _coordinates(place)
            results.append(
                {
                    "name": place.get("display_name", "Unknown place"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "type": place.get("type"),
                    "address": place.get("display_name"),
                    "map_url": (
                        "https://www.openstreetmap.org/?mlat="
                        f"{latitude}&mlon={longitude}#map=16/{latitude}/{longitude}"
                    ),
                }
            )
        return {"query": query, "near": near.strip(), "results": results, "source": "OpenStreetMap"}

    return _result("Place search", operation)


def directions(origin: str, destination: str, mode: str) -> InformationResult:
    def operation() -> dict:
        start = origin.strip()
        end = destination.strip()
        if not start or not end:
            raise ValueError("Both an origin and destination are required.")
        profiles = {"driving": "driving", "walking": "foot", "cycling": "bike"}
        normalized_mode = mode.strip().casefold() or "driving"
        if normalized_mode not in profiles:
            raise ValueError("Mode must be driving, walking, or cycling.")
        start_place = _geocode(start, 1)[0]
        end_place = _geocode(end, 1)[0]
        start_lon, start_lat = _coordinates(start_place)
        end_lon, end_lat = _coordinates(end_place)
        route_url = (
            f"https://router.project-osrm.org/route/v1/{profiles[normalized_mode]}/"
            f"{start_lon},{start_lat};{end_lon},{end_lat}?"
            + urlencode({"overview": "false", "steps": "false"})
        )
        payload = _request_json(route_url, "OpenStreetMap routing")
        routes = payload.get("routes") if isinstance(payload, dict) else None
        if not isinstance(routes, list) or not routes:
            raise ValueError("No route was found between those locations.")
        route = routes[0]
        distance_m = route.get("distance")
        duration_s = route.get("duration")
        if not isinstance(distance_m, (int, float)) or not isinstance(duration_s, (int, float)):
            raise ValueError("The routing service returned incomplete route data.")
        return {
            "origin": _location_label(start_place),
            "destination": _location_label(end_place),
            "mode": normalized_mode,
            "distance_km": round(distance_m / 1000, 1),
            "duration_minutes": round(duration_s / 60),
            "source": "OpenStreetMap/OSRM",
        }

    return _result("Directions", operation)


def web_search(query: str, max_results: int) -> InformationResult:
    def operation() -> dict:
        topic = query.strip()
        if not topic:
            raise ValueError("A web-search query is required.")
        url = "https://html.duckduckgo.com/html/?" + urlencode({"q": topic})
        raw = _request_bytes(url, "Web search", "text/html")
        parser = _SearchParser()
        parser.feed(raw.decode("utf-8", errors="replace"))
        results = parser.items[:max_results]
        if not results:
            raise ValueError(f"No web results found for '{topic}'.")
        return {"query": topic, "results": results, "source": "DuckDuckGo"}

    return _result("Web search", operation)


def travel_search(destination: str, topic: str, max_results: int) -> InformationResult:
    query = f"{topic.strip() or 'travel guide and current visitor information'} {destination.strip()}".strip()
    return web_search(query, max_results)


def game_search(query: str, max_results: int) -> InformationResult:
    return web_search(f"{query.strip()} gaming guide latest update", max_results)