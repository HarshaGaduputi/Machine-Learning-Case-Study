"""Conservative input contracts shared by the web interface and smoke tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def present(value: object) -> bool:
    return value is not None and bool(pd.notna(value))


def outside_observed(values: dict, bounds: dict) -> list[str]:
    return [
        field for field, value in values.items()
        if field in bounds and present(value)
        and not bounds[field]["min"] <= float(value) <= bounds[field]["max"]
    ]


def regression_issues(values: dict, evidence: dict, numeric_fields: list[str]) -> list[str]:
    issues = []
    count = sum(present(values.get(name)) for name in numeric_fields)
    required = evidence["numeric_inputs_required"]
    if count < required:
        issues.append(f"Enter at least {required} of the {len(numeric_fields)} numeric building and weather fields; {count} are supplied.")
    unusual = outside_observed(values, evidence["bounds"])
    if unusual:
        issues.append("Outside the source data's observed range: " + ", ".join(unusual) + ".")
    return issues


def classification_issues(values: dict, evidence: dict, measured_fields: list[str]) -> list[str]:
    issues = []
    count = sum(present(values.get(name)) for name in measured_fields)
    required = evidence["measured_fields_required"]
    if count < required:
        issues.append(f"Enter at least {required} measured water fields; {count} of {len(measured_fields)} are supplied.")
    unusual = outside_observed(values, evidence["bounds"])
    if unusual:
        issues.append("Outside the source data's observed range: " + ", ".join(unusual) + ".")
    year = values.get("sample_year")
    year_bounds = evidence["sample_year_bounds"]
    if present(year) and not year_bounds["min"] <= float(year) <= year_bounds["max"]:
        issues.append("Sample year is outside the dated records used to train the model.")
    return issues


def clustering_issues(values: dict, evidence: dict, distance: float) -> list[str]:
    issues = []
    groups = [
        ("pollutant", ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"], "minimum_pollutants"),
        ("weather", ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"], "minimum_weather"),
        ("time and wind", ["month", "hour", "wd"], "minimum_time_fields"),
    ]
    for label, fields, key in groups:
        count = sum(present(values.get(field)) for field in fields)
        if count < evidence[key]:
            issues.append(f"Enter at least {evidence[key]} {label} fields; {count} are supplied.")
    unusual = outside_observed(values, evidence["bounds"])
    if unusual:
        issues.append("Outside the Beijing source data's observed range: " + ", ".join(unusual) + ".")
    if not np.isfinite(distance) or distance > evidence["distance_p99"]:
        issues.append("This combination is unusually far from historical cluster centres. Check the readings and units.")
    return issues


def result_record(*, task: str, model_id: str, values: dict, imputed_fields: list[str], result: dict, limits: list[str]) -> str:
    clean_values = {}
    for key, value in values.items():
        if not present(value):
            clean_values[key] = None
        elif isinstance(value, np.generic):
            clean_values[key] = value.item()
        else:
            clean_values[key] = value
    return json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task,
        "model_id": model_id,
        "entered_values": clean_values,
        "imputed_fields": imputed_fields,
        "result": result,
        "limitations": limits,
    }, indent=2, allow_nan=False)
