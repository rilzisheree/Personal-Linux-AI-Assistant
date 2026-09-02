from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from local_ai_assistant import information_tools


class InformationToolsTests(unittest.TestCase):
    def test_weather_uses_resolved_location_and_requested_forecast_day(self) -> None:
        forecast = {
            "timezone": "Asia/Riyadh",
            "current": {
                "temperature_2m": 31,
                "apparent_temperature": 34,
                "relative_humidity_2m": 40,
                "precipitation": 0,
                "weather_code": 1,
                "wind_speed_10m": 12,
            },
            "daily": {
                "time": ["2026-09-02", "2026-09-03"],
                "weather_code": [1, 61],
                "temperature_2m_max": [35, 33],
                "temperature_2m_min": [27, 26],
                "precipitation_probability_max": [0, 70],
                "sunrise": ["2026-09-02T05:50", "2026-09-03T05:49"],
                "sunset": ["2026-09-02T18:20", "2026-09-03T18:19"],
            },
        }
        with patch(
            "local_ai_assistant.information_tools._geocode",
            return_value=[
                {
                    "lat": "21.5433",
                    "lon": "39.1728",
                    "display_name": "Jeddah, Saudi Arabia",
                    "address": {"city": "Jeddah", "country": "Saudi Arabia"},
                }
            ],
        ), patch(
            "local_ai_assistant.information_tools._request_json",
            return_value=forecast,
        ):
            result = information_tools.weather("Jeddah", "tomorrow", "metric")

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertEqual(payload["location"], "Jeddah, Saudi Arabia")
        self.assertEqual(payload["requested_date"], "2026-09-03")
        self.assertEqual(payload["forecast"]["condition"], "slight rain")

    def test_news_parses_dated_rss_results(self) -> None:
        root = ElementTree.fromstring(
            """<rss><channel><item>
            <title>Important headline</title>
            <link>https://example.com/news</link>
            <pubDate>Wed, 02 Sep 2026 10:00:00 GMT</pubDate>
            <description>&lt;p&gt;A short summary.&lt;/p&gt;</description>
            </item></channel></rss>"""
        )
        with patch(
            "local_ai_assistant.information_tools._request_xml",
            return_value=root,
        ):
            result = information_tools.news("Saudi Arabia", 5)

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertEqual(payload["results"][0]["title"], "Important headline")
        self.assertEqual(payload["results"][0]["summary"], "A short summary.")

    def test_knowledge_search_adds_wikipedia_summary_when_available(self) -> None:
        with patch(
            "local_ai_assistant.information_tools._request_json",
            side_effect=[
                {"query": {"search": [{"title": "Napoleon", "snippet": "<b>French</b> leader"}]}},
                {"extract": "Napoleon was a French military leader."},
            ],
        ):
            result = information_tools.knowledge_search("Napoleon", 3)

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertEqual(payload["results"][0]["summary"], "Napoleon was a French military leader.")
        self.assertEqual(payload["results"][0]["snippet"], "French leader")

    def test_currency_returns_current_rate_data(self) -> None:
        with patch(
            "local_ai_assistant.information_tools._request_json",
            return_value={
                "result": "success",
                "base_code": "SAR",
                "time_last_update_utc": "Wed, 02 Sep 2026 00:02:31 +0000",
                "rates": {"USD": 0.267},
            },
        ):
            result = information_tools.currency(500, "SAR", "USD")

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertEqual(payload["converted_amount"], 133.5)
        self.assertEqual(payload["rate"], 0.267)
        self.assertIn("02 Sep 2026", payload["rate_date"])

    def test_directions_geocode_both_ends_and_returns_distance(self) -> None:
        places = [
            {
                "lat": "21.5433",
                "lon": "39.1728",
                "display_name": "Jeddah, Saudi Arabia",
                "address": {"city": "Jeddah", "country": "Saudi Arabia"},
            },
            {
                "lat": "21.2703",
                "lon": "40.4158",
                "display_name": "Taif, Saudi Arabia",
                "address": {"city": "Taif", "country": "Saudi Arabia"},
            },
        ]
        with patch(
            "local_ai_assistant.information_tools._geocode",
            side_effect=[[places[0]], [places[1]]],
        ), patch(
            "local_ai_assistant.information_tools._request_json",
            return_value={"routes": [{"distance": 180000, "duration": 7200}]},
        ):
            result = information_tools.directions("Jeddah", "Taif", "driving")

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertEqual(payload["distance_km"], 180)
        self.assertEqual(payload["duration_minutes"], 120)
        self.assertEqual(payload["mode"], "driving")

    def test_network_failure_is_returned_as_a_failed_tool_result(self) -> None:
        with patch(
            "local_ai_assistant.information_tools._request_json",
            side_effect=ValueError("service timed out"),
        ):
            result = information_tools.currency(1, "USD", "SAR")

        self.assertFalse(result.success)
        self.assertIn("Currency conversion failed", result.content)
        self.assertIn("service timed out", result.content)


if __name__ == "__main__":
    unittest.main()