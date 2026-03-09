#!/usr/bin/env python3
"""Fetch full NCAA MBB history from NatStat API v3.5 and write parquet files.

Outputs:
- games.parquet
- teams.parquet
- team_stats.parquet
- elo.parquet

API key handling:
- Preferred: export NATSTAT_API_KEY=xxxx-xxxxxx
- Optional: pass --api-key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests


BASE_URL = "https://api3.natst.at"
DEFAULT_LEVEL = "mbb"
DEFAULT_START_SEASON = 2008


class NatStatError(RuntimeError):
    """Raised when the API returns a structured error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch full-history NatStat NCAA MBB data and write parquet files."
    )
    parser.add_argument("--api-key", default=None, help="NatStat API key. Defaults to NATSTAT_API_KEY env var.")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help="NatStat level code (default: mbb).")
    parser.add_argument("--start-season", type=int, default=None, help="Force first season year (inclusive).")
    parser.add_argument("--end-season", type=int, default=None, help="Force last season year (inclusive).")
    parser.add_argument("--output-dir", default=".", help="Directory for parquet outputs.")
    parser.add_argument("--sleep-seconds", type=float, default=0.15, help="Delay between API calls.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0, help="HTTP timeout.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per request.")
    return parser.parse_args()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    # Normalize nested endpoint payloads so parquet conversion is stable.
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].map(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
        )
    return df


def build_url(api_key: str, endpoint: str, level: str, range_part: str | None, offset: int | None) -> str:
    parts = [BASE_URL.rstrip("/"), api_key, endpoint, level]
    if range_part:
        parts.append(range_part)
    if offset and offset > 0:
        parts.append(str(offset))
    return "/".join(parts)


def _extract_records_from_payload(payload: Any, endpoint: str) -> list[dict[str, Any]]:
    def normalize_records(container: Any) -> list[dict[str, Any]]:
        if isinstance(container, list):
            return [r for r in container if isinstance(r, dict)]
        if isinstance(container, dict):
            # NatStat often returns endpoint maps keyed like game_12345, team_ABC, etc.
            values = list(container.values())
            if values and all(isinstance(v, dict) for v in values):
                return [v for v in values if isinstance(v, dict)]
            if all(not isinstance(v, (list, dict)) for v in values):
                return [container]
        return []

    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]

    if not isinstance(payload, dict):
        return []

    success = payload.get("success")
    if success == 0:
        err = payload.get("error") or payload.get("meta") or "Unknown API error"
        raise NatStatError(str(err))

    candidates: list[Any] = []
    for key in ("data", "results", endpoint, endpoint.rstrip("s"), "items"):
        if key in payload:
            candidates.append(payload[key])

    # Fallback: first list/map-like value in payload.
    if not candidates:
        for value in payload.values():
            if isinstance(value, (list, dict)):
                candidates.append(value)

    for candidate in candidates:
        records = normalize_records(candidate)
        if records:
            return records

    return []


def request_json(session: requests.Session, url: str, timeout_seconds: float, max_retries: int) -> Any:
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout_seconds)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries - 1:
                raise NatStatError(f"Request failed after {max_retries} attempts: {url} :: {exc}") from exc
            sleep_for = min(8.0, 0.6 * (2 ** attempt))
            time.sleep(sleep_for)
    raise NatStatError("Unreachable retry loop")


def fetch_paged_records(
    session: requests.Session,
    api_key: str,
    endpoint: str,
    level: str,
    range_part: str | None,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    page_size = 500 if endpoint in {"events", "playbyplay", "pitchfx", "teamcodes", "leaguecodes"} else 100
    all_records: list[dict[str, Any]] = []

    # Protect against pathological pagination loops by tracking page fingerprints.
    seen_page_signatures: set[tuple[Any, ...]] = set()

    for offset in range(0, 200_000, page_size):
        url = build_url(api_key=api_key, endpoint=endpoint, level=level, range_part=range_part, offset=offset)
        payload = request_json(
            session=session,
            url=url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        records = _extract_records_from_payload(payload, endpoint=endpoint)
        if not records:
            break

        signature = tuple(sorted((str(r.get("id", "")), str(r.get("code", ""))) for r in records[:5]))
        if signature in seen_page_signatures:
            break
        seen_page_signatures.add(signature)

        all_records.extend(records)

        if len(records) < page_size:
            break

        time.sleep(sleep_seconds)

    return all_records


def extract_seasons(records: Iterable[dict[str, Any]]) -> list[int]:
    seasons: set[int] = set()
    for row in records:
        for value in row.values():
            if isinstance(value, int) and 1900 <= value <= 2100:
                seasons.add(value)
            elif isinstance(value, str) and value.isdigit() and len(value) == 4:
                year = int(value)
                if 1900 <= year <= 2100:
                    seasons.add(year)
    return sorted(seasons)


def infer_available_seasons(
    session: requests.Session,
    api_key: str,
    level: str,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> list[int]:
    try:
        records = fetch_paged_records(
            session=session,
            api_key=api_key,
            endpoint="season",
            level=level,
            range_part=None,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        seasons = extract_seasons(records)
        if seasons:
            return seasons
    except NatStatError:
        pass

    current_year = datetime.utcnow().year
    return list(range(DEFAULT_START_SEASON, current_year + 1))


def fetch_season_df(
    session: requests.Session,
    api_key: str,
    endpoint: str,
    level: str,
    season: int,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
    extra_range_suffix: str | None = None,
) -> pd.DataFrame:
    range_part = str(season)
    if extra_range_suffix:
        range_part = f"{range_part},{extra_range_suffix}"

    records = fetch_paged_records(
        session=session,
        api_key=api_key,
        endpoint=endpoint,
        level=level,
        range_part=range_part,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df["season"] = season
    return clean_df(df)


def dedupe_rows(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in subset if c in df.columns]
    if not cols:
        return df.drop_duplicates().reset_index(drop=True)
    return df.drop_duplicates(subset=cols).reset_index(drop=True)


def _as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_div(n: float, d: float) -> float | None:
    return round(n / d, 4) if d else None


def build_team_stats_from_teamperfs(
    session: requests.Session,
    api_key: str,
    level: str,
    season: int,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    records = fetch_paged_records(
        session=session,
        api_key=api_key,
        endpoint="teamperfs",
        level=level,
        range_part=str(season),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    if not records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for rec in records:
        stats = rec.get("stats", {}) if isinstance(rec, dict) else {}
        game = rec.get("game", {}) if isinstance(rec, dict) else {}
        team_code = rec.get("team-code")
        team_name = rec.get("team-name")
        winner_code = game.get("winner-code") if isinstance(game, dict) else None
        win_or_loss = game.get("winorloss") if isinstance(game, dict) else None
        is_win = None
        if isinstance(win_or_loss, str) and win_or_loss.upper() in {"W", "L"}:
            is_win = 1.0 if win_or_loss.upper() == "W" else 0.0
        elif team_code and winner_code:
            is_win = 1.0 if team_code == winner_code else 0.0
        else:
            is_win = 0.0
        rows.append(
            {
                "team_code": team_code,
                "team_name": team_name,
                "game_id": game.get("id") if isinstance(game, dict) else None,
                "pts": _as_float(stats.get("pts")),
                "fgm": _as_float(stats.get("fgm")),
                "fga": _as_float(stats.get("fga")),
                "threefm": _as_float(stats.get("threefm")),
                "threefa": _as_float(stats.get("threefa")),
                "ftm": _as_float(stats.get("ftm")),
                "fta": _as_float(stats.get("fta")),
                "reb": _as_float(stats.get("reb")),
                "oreb": _as_float(stats.get("oreb")),
                "ast": _as_float(stats.get("ast")),
                "stl": _as_float(stats.get("stl")),
                "blk": _as_float(stats.get("blk")),
                "to": _as_float(stats.get("to")),
                "f": _as_float(stats.get("f")),
                "min": _as_float(stats.get("min")),
                "w": is_win,
            }
        )

    perf = pd.DataFrame(rows)
    if perf.empty:
        return pd.DataFrame()

    grouped = (
        perf.groupby(["team_code", "team_name"], dropna=False)
        .agg(
            g=("game_id", "nunique"),
            min=("min", "sum"),
            fga=("fga", "sum"),
            fgm=("fgm", "sum"),
            threefa=("threefa", "sum"),
            threefm=("threefm", "sum"),
            fta=("fta", "sum"),
            ftm=("ftm", "sum"),
            reb=("reb", "sum"),
            oreb=("oreb", "sum"),
            ast=("ast", "sum"),
            stl=("stl", "sum"),
            blk=("blk", "sum"),
            to=("to", "sum"),
            f=("f", "sum"),
            pts=("pts", "sum"),
            w=("w", "sum"),
        )
        .reset_index()
    )

    grouped["l"] = grouped["g"] - grouped["w"]
    grouped["winpct"] = grouped.apply(lambda r: _safe_div(float(r["w"]), float(r["g"])), axis=1)
    grouped["2fa"] = grouped["fga"] - grouped["threefa"]
    grouped["2fm"] = grouped["fgm"] - grouped["threefm"]
    grouped["3fa"] = grouped["threefa"]
    grouped["3fm"] = grouped["threefm"]
    grouped["apg"] = grouped.apply(lambda r: _safe_div(float(r["ast"]), float(r["g"])), axis=1)
    grouped["bpg"] = grouped.apply(lambda r: _safe_div(float(r["blk"]), float(r["g"])), axis=1)
    grouped["spg"] = grouped.apply(lambda r: _safe_div(float(r["stl"]), float(r["g"])), axis=1)
    grouped["topg"] = grouped.apply(lambda r: _safe_div(float(r["to"]), float(r["g"])), axis=1)
    grouped["fpg"] = grouped.apply(lambda r: _safe_div(float(r["f"]), float(r["g"])), axis=1)
    grouped["rpg"] = grouped.apply(lambda r: _safe_div(float(r["reb"]), float(r["g"])), axis=1)
    grouped["season"] = season

    cols = [
        "team_name",
        "team_code",
        "2fa",
        "2fm",
        "3fa",
        "3fm",
        "apg",
        "ast",
        "blk",
        "bpg",
        "f",
        "fga",
        "fgm",
        "fpg",
        "fta",
        "ftm",
        "g",
        "l",
        "min",
        "oreb",
        "pts",
        "reb",
        "rpg",
        "spg",
        "stl",
        "to",
        "topg",
        "w",
        "winpct",
        "season",
    ]
    out = grouped[cols].copy()
    return clean_df(out)


def main() -> int:
    args = parse_args()

    api_key = args.api_key or os.environ.get("NATSTAT_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set NATSTAT_API_KEY.", file=sys.stderr)
        return 2

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "natstat-full-history-fetcher/1.0"})

    seasons = infer_available_seasons(
        session=session,
        api_key=api_key,
        level=args.level,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )

    if args.start_season is not None:
        seasons = [y for y in seasons if y >= args.start_season]
    if args.end_season is not None:
        seasons = [y for y in seasons if y <= args.end_season]

    if not seasons:
        print("ERROR: No seasons selected after filters.", file=sys.stderr)
        return 3

    print(f"Using seasons: {min(seasons)}..{max(seasons)} ({len(seasons)} total)")

    games_frames: list[pd.DataFrame] = []
    stats_frames: list[pd.DataFrame] = []
    elo_frames: list[pd.DataFrame] = []
    teams_frames: list[pd.DataFrame] = []

    for season in seasons:
        print(f"Fetching season {season}...")

        games_df = fetch_season_df(
            session=session,
            api_key=api_key,
            endpoint="games",
            level=args.level,
            season=season,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        if not games_df.empty:
            games_frames.append(games_df)

        stats_df = fetch_season_df(
            session=session,
            api_key=api_key,
            endpoint="stats",
            level=args.level,
            season=season,
            extra_range_suffix="team",
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        if stats_df.empty:
            stats_df = build_team_stats_from_teamperfs(
                session=session,
                api_key=api_key,
                level=args.level,
                season=season,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )
        if not stats_df.empty:
            stats_frames.append(stats_df)

        elo_df = fetch_season_df(
            session=session,
            api_key=api_key,
            endpoint="elo",
            level=args.level,
            season=season,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        if not elo_df.empty:
            keep_cols = [c for c in ("team", "code", "elo", "elorank", "season") if c in elo_df.columns]
            if keep_cols:
                elo_df = elo_df[keep_cols]
            elo_frames.append(elo_df)

        teams_df = fetch_season_df(
            session=session,
            api_key=api_key,
            endpoint="teams",
            level=args.level,
            season=season,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        if not teams_df.empty:
            teams_frames.append(teams_df)

    # Combine + dedupe.
    games = dedupe_rows(pd.concat(games_frames, ignore_index=True) if games_frames else pd.DataFrame(), ["id", "season"])
    team_stats = dedupe_rows(pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame(), ["team_code", "season"])
    elo = dedupe_rows(pd.concat(elo_frames, ignore_index=True) if elo_frames else pd.DataFrame(), ["code", "season"])

    teams_raw = pd.concat(teams_frames, ignore_index=True) if teams_frames else pd.DataFrame()
    if not teams_raw.empty:
        preferred_cols = [c for c in ("code", "name", "location") if c in teams_raw.columns]
        teams = teams_raw[preferred_cols] if preferred_cols else teams_raw
        teams = dedupe_rows(teams, ["code"])
    else:
        teams = pd.DataFrame(columns=["code", "name", "location"])

    # Write outputs.
    games_path = os.path.join(output_dir, "games.parquet")
    teams_path = os.path.join(output_dir, "teams.parquet")
    team_stats_path = os.path.join(output_dir, "team_stats.parquet")
    elo_path = os.path.join(output_dir, "elo.parquet")

    games.to_parquet(games_path, index=False)
    teams.to_parquet(teams_path, index=False)
    team_stats.to_parquet(team_stats_path, index=False)
    elo.to_parquet(elo_path, index=False)

    print("Wrote files:")
    print(f"- {games_path} rows={len(games)}")
    print(f"- {teams_path} rows={len(teams)}")
    print(f"- {team_stats_path} rows={len(team_stats)}")
    print(f"- {elo_path} rows={len(elo)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
