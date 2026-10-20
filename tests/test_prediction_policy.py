"""Focused checks for refusal rules and the exported result contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.prediction_policy import (
    classification_issues, clustering_issues, outside_observed,
    regression_issues, result_record,
)
from app.dashboard import parse_water

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads((ROOT / "results/release_validation.json").read_text())


class PredictionPolicyTests(unittest.TestCase):
    def test_building_requires_source_like_completeness(self):
        evidence = EVIDENCE["regression"]
        fields = list(evidence["bounds"])
        values = {name: evidence["bounds"][name]["p01"] for name in fields[:8]}
        self.assertEqual(regression_issues(values, evidence, fields), [])
        values.pop(fields[0])
        self.assertTrue(any("at least 8" in issue for issue in regression_issues(values, evidence, fields)))
        values[fields[0]] = evidence["bounds"][fields[0]]["max"] + 1
        self.assertTrue(any("observed range" in issue for issue in regression_issues(values, evidence, fields)))

    def test_water_requires_measured_values_and_in_range_date(self):
        evidence = EVIDENCE["classification"]
        fields = list(evidence["bounds"])
        values = {name: evidence["bounds"][name]["p01"] for name in fields[:12]}
        self.assertEqual(classification_issues(values, evidence, fields), [])
        values.pop(fields[0])
        self.assertTrue(any("at least 12" in issue for issue in classification_issues(values, evidence, fields)))
        values[fields[0]] = evidence["bounds"][fields[0]]["p01"]
        values["sample_year"] = evidence["sample_year_bounds"]["max"] + 1
        self.assertTrue(any("Sample year" in issue for issue in classification_issues(values, evidence, fields)))

    def test_air_rejects_sparse_and_distant_input(self):
        evidence = EVIDENCE["clustering"]
        values = {
            **{name: evidence["bounds"][name]["p01"] for name in ["PM2.5", "PM10", "SO2", "NO2", "TEMP", "PRES", "DEWP"]},
            "month": 3, "hour": 12,
        }
        self.assertEqual(clustering_issues(values, evidence, 2.0), [])
        values.pop("NO2")
        self.assertTrue(any("pollutant" in issue for issue in clustering_issues(values, evidence, 2.0)))
        values["NO2"] = evidence["bounds"]["NO2"]["p01"]
        self.assertTrue(any("cluster centres" in issue for issue in clustering_issues(values, evidence, 100.0)))

    def test_air_checks_input_before_model_distance(self):
        evidence = EVIDENCE["clustering"]
        values = {
            **{name: evidence["bounds"][name]["p01"] for name in
               ["PM2.5", "PM10", "SO2", "NO2", "TEMP", "PRES", "DEWP"]},
            "month": 3, "hour": 12,
        }
        self.assertEqual(clustering_issues(values, evidence), [])
        values["PM2.5"] = float("inf")
        self.assertIn("PM2.5", outside_observed(values, evidence["bounds"]))
        self.assertTrue(clustering_issues(values, evidence))

    def test_air_uses_distance_limit_matching_input_completeness(self):
        evidence = dict(EVIDENCE["clustering"])
        evidence.update({"distance_p99_complete": 4.0, "distance_p99_masked": 6.0})
        values = {
            **{name: evidence["bounds"][name]["p01"] for name in
               ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]},
            "month": 3, "hour": 12, "wd": "N",
        }
        self.assertTrue(any("cluster centres" in issue
                            for issue in clustering_issues(values, evidence, 5.0)))
        values["CO"] = None
        self.assertEqual(clustering_issues(values, evidence, 5.0), [])

    def test_water_parser_rejects_oversized_value(self):
        values, errors = parse_water({"pH": "1" * 33})
        self.assertEqual(values, {})
        self.assertEqual(errors, ["pH"])

    def test_download_record_is_versioned_and_valid_json(self):
        record = json.loads(result_record(
            task="example", model_id="abc123", values={"observed": np.float64(1.5), "missing": np.nan},
            imputed_fields=["missing"], result={"value": 2}, limits=["Research output"],
        ))
        self.assertEqual(record["model_id"], "abc123")
        self.assertIsNone(record["entered_values"]["missing"])
        self.assertEqual(record["imputed_fields"], ["missing"])
        self.assertIn("generated_at_utc", record)


if __name__ == "__main__":
    unittest.main()
