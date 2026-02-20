# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Shared Open-Meteo Archive API fetch + cache utility.

All neuralSpring experiments that need real weather data use this module
to avoid duplicating retry/cache logic across scripts.

Data source: Open-Meteo Archive API (ERA5 reanalysis)
  URL:     https://archive-api.open-meteo.com/v1/archive
  Auth:    None required (free tier, 10k requests/day)
  License: CC BY 4.0
  Docs:    https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

API_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS_BASIC = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
]

DAILY_VARS_SOLAR = DAILY_VARS_BASIC + ["shortwave_radiation_sum"]

DAILY_VARS_ET0 = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
]

# Well-known locations used across neuralSpring experiments
LOCATIONS = {
    "east_lansing_mi": {"lat": 42.73, "lon": -84.48, "alt": 256, "tz": "America/Detroit"},
    "las_cruces_nm": {"lat": 32.35, "lon": -106.74, "alt": 1200, "tz": "America/Denver"},
    "davis_ca": {"lat": 38.53, "lon": -121.74, "alt": 16, "tz": "America/Los_Angeles"},
}

DEFAULT_PERIOD = ("2020-01-01", "2023-12-31")

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "weather"


def _cache_key(lat: float, lon: float, start: str, end: str, variables: list[str] | None) -> str:
    var_tag = ",".join(sorted(variables)) if variables else "default"
    tag = f"{lat:.2f}_{lon:.2f}_{start}_{end}_{var_tag}"
    return hashlib.md5(tag.encode()).hexdigest()[:12]


def fetch_daily(
    lat: float,
    lon: float,
    start: str,
    end: str,
    *,
    timezone: str = "America/Detroit",
    include_solar: bool = False,
    variables: list[str] | None = None,
    retries: int = 3,
) -> dict[str, np.ndarray | list[str]]:
    """Fetch daily weather from the Open-Meteo Archive API.

    If *variables* is provided, those exact API variable names are fetched
    and returned under their short-name keys (see VAR_SHORT_NAMES).
    Otherwise, returns: date, tmax, tmin, precip, wind, humidity, and
    optionally solar (MJ/m2/day) if include_solar=True.

    Raises requests.HTTPError or ConnectionError after exhausting retries.
    """
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' package required for Open-Meteo API.", file=sys.stderr)
        raise

    if variables is None:
        variables = DAILY_VARS_SOLAR if include_solar else DAILY_VARS_BASIC

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(variables),
        "timezone": timezone,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == retries - 1:
                raise
            print(f"  Open-Meteo attempt {attempt + 1} failed: {exc}, retrying...")
            time.sleep(2**attempt)

    data = resp.json()
    daily = data["daily"]

    result: dict[str, np.ndarray | list[str]] = {"date": daily["time"]}
    for var in variables:
        short = VAR_SHORT_NAMES.get(var, var)
        raw = daily[var]
        arr = np.array(raw, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        result[short] = arr

    return result


VAR_SHORT_NAMES = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "precip",
    "wind_speed_10m_max": "wind",
    "relative_humidity_2m_mean": "humidity",
    "relative_humidity_2m_max": "rhmax",
    "relative_humidity_2m_min": "rhmin",
    "shortwave_radiation_sum": "solar",
}


def load_or_fetch(
    lat: float,
    lon: float,
    start: str = DEFAULT_PERIOD[0],
    end: str = DEFAULT_PERIOD[1],
    *,
    timezone: str = "America/Detroit",
    include_solar: bool = False,
    variables: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, np.ndarray | list[str]]:
    """Load from cache if available, otherwise fetch from API and cache.

    Cache is a .npz file keyed by location + date range + variable set.
    If *variables* is provided, it takes precedence over *include_solar*.
    """
    if variables is None and include_solar:
        variables = DAILY_VARS_SOLAR

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(lat, lon, start, end, variables)
    cache_path = cache_dir / f"open_meteo_{key}.npz"

    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        result: dict[str, np.ndarray | list[str]] = {k: loaded[k] for k in loaded.files}
        if "date" in result:
            result["date"] = result["date"].tolist()
        n = max((len(v) for v in result.values() if isinstance(v, np.ndarray)), default=0)
        print(f"  Loaded cached weather: {cache_path.name} ({n} days)")
        return result

    print(f"  Fetching Open-Meteo: ({lat}, {lon}) {start} to {end}...")
    result = fetch_daily(lat, lon, start, end, timezone=timezone, variables=variables)

    save_kwargs = {}
    for k, v in result.items():
        save_kwargs[k] = np.array(v) if isinstance(v, list) else v

    np.savez_compressed(cache_path, **save_kwargs)
    n = max((len(v) for v in result.values() if isinstance(v, np.ndarray)), default=0)
    print(f"  Cached to {cache_path.name} ({n} days)")

    return result


def load_or_fetch_location(
    name: str,
    start: str = DEFAULT_PERIOD[0],
    end: str = DEFAULT_PERIOD[1],
    *,
    include_solar: bool = False,
    variables: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, np.ndarray | list[str]]:
    """Convenience: fetch by well-known location name (e.g. 'east_lansing_mi')."""
    loc = LOCATIONS[name]
    return load_or_fetch(
        loc["lat"],
        loc["lon"],
        start,
        end,
        timezone=loc["tz"],
        include_solar=include_solar,
        variables=variables,
        cache_dir=cache_dir,
    )


def generate_synthetic_weather(n_days: int = 1461, seed: int = 42) -> dict:
    """Fallback: synthetic Michigan weather (4 years) when API is unavailable."""
    rng = np.random.default_rng(seed)
    doy = np.arange(n_days) % 365

    seasonal_tmax = 8.5 + 15.0 * np.sin(2 * np.pi * (doy - 100) / 365)
    noise = np.zeros(n_days)
    noise[0] = rng.normal(0, 3)
    for i in range(1, n_days):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 3) * 0.71

    tmax = seasonal_tmax + noise
    tmin = tmax - 10 + rng.normal(0, 1.5, n_days)
    tmin = np.minimum(tmin, tmax - 2).astype(np.float32)
    tmax = tmax.astype(np.float32)

    precip = np.where(rng.random(n_days) < 0.35, rng.exponential(6, n_days), 0).astype(np.float32)
    wind = (8 + 5 * rng.standard_normal(n_days)).clip(0.5, 40).astype(np.float32)
    humidity = (
        (65 + 15 * np.sin(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 8, n_days))
        .clip(20, 100)
        .astype(np.float32)
    )

    return {
        "date": [f"synth-day-{i}" for i in range(n_days)],
        "tmax": tmax,
        "tmin": tmin,
        "precip": precip,
        "wind": wind,
        "humidity": humidity,
    }
