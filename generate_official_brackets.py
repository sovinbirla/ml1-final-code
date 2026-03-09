#!/usr/bin/env python3
"""
Run the forecasting pipeline and render official-style NCAA bracket PNGs
for the men's and women's projections using the latest non-tournament results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts_modeling"
OUT.mkdir(exist_ok=True)

ROUND_ORDER = [
    "First Four",
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final 4",
    "Championship",
]

MAX_SLOT_ROWS = 64


def _parse_gameday(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _latest_gameday_for_season(games_file: Path, season: int) -> pd.Timestamp:
    games = pd.read_parquet(games_file).copy()
    games["gameday"] = _parse_gameday(games["gameday"])
    games = games[
        (games["season"] == season)
        & games["gameday"].notna()
        & pd.to_numeric(games["score_home"], errors="coerce").notna()
        & pd.to_numeric(games["score_vis"], errors="coerce").notna()
    ].copy()
    if games.empty:
        raise ValueError(f"No completed games found for season={season} in {games_file}")
    games["score_home"] = pd.to_numeric(games["score_home"], errors="coerce")
    games["score_vis"] = pd.to_numeric(games["score_vis"], errors="coerce")
    games = games.dropna(subset=["score_home", "score_vis"])
    mm_proxy = (
        games["league"].isna() & (games["gameday"].dt.month == 3) & (games["gameday"].dt.day >= 15)
    ) | (
        games["league"].isna() & (games["gameday"].dt.month == 4) & (games["gameday"].dt.day <= 15)
    )
    games = games.loc[~mm_proxy].copy()
    if games.empty:
        raise ValueError(f"No non-tournament completed games found for season={season} in {games_file}")
    return games["gameday"].max().normalize()


def _load_team_lookup(teams_file: Path, fallback_file: Path, season: int) -> pd.DataFrame:
    t = pd.read_parquet(teams_file).copy()
    if "code" not in t.columns or "name" not in t.columns:
        t = pd.DataFrame(columns=["team_code", "team_name"])
    else:
        t = t[["code", "name"]].rename(columns={"code": "team_code", "name": "team_name"})
        if "season" in t.columns and pd.api.types.is_numeric_dtype(t["season"]):
            if t["season"].notna().any():
                t = t[(t["season"].isna()) | (t["season"] == season)]
        t = t.drop_duplicates("team_code")

    if fallback_file.exists():
        fb = pd.read_parquet(fallback_file)
        if {"team_code", "team_name"}.issubset(fb.columns):
            fb = fb[["team_code", "team_name"]].drop_duplicates("team_code")
            t = (
                t.merge(fb, on="team_code", how="outer", suffixes=("", "_fb"))
                .assign(team_name=lambda d: d["team_name"].fillna(d["team_name_fb"]))
                .loc[:, ["team_code", "team_name"]]
                .drop_duplicates("team_code")
            )
    return t


def build_selection_features_live(games_file: Path, elo_file: Path, season: int, league_name: str) -> pd.DataFrame:
    games = pd.read_parquet(games_file).copy()
    games["gameday"] = _parse_gameday(games["gameday"])
    as_of = _latest_gameday_for_season(games_file, season)

    games = games[
        (games["season"] == season)
        & (games["gameday"].notna())
        & (games["gameday"] <= as_of)
        & pd.to_numeric(games["score_home"], errors="coerce").notna()
        & pd.to_numeric(games["score_vis"], errors="coerce").notna()
    ].copy()
    games["score_home"] = pd.to_numeric(games["score_home"], errors="coerce")
    games["score_vis"] = pd.to_numeric(games["score_vis"], errors="coerce")
    games = games.dropna(subset=["score_home", "score_vis"])

    mm_proxy = (
        games["league"].isna() & (games["gameday"].dt.month == 3) & (games["gameday"].dt.day >= 15)
    ) | (
        games["league"].isna() & (games["gameday"].dt.month == 4) & (games["gameday"].dt.day <= 15)
    )
    games = games.loc[~mm_proxy].copy()

    home = pd.DataFrame(
        {
            "season": games["season"],
            "gameday": games["gameday"],
            "team_code": games["home_code"],
            "opp_code": games["visitor_code"],
            "conference_proxy": games["league"].fillna("UNKNOWN"),
            "win": (games["winner_code"] == games["home_code"]).astype(float),
        }
    )
    vis = pd.DataFrame(
        {
            "season": games["season"],
            "gameday": games["gameday"],
            "team_code": games["visitor_code"],
            "opp_code": games["home_code"],
            "conference_proxy": games["league"].fillna("UNKNOWN"),
            "win": (games["winner_code"] == games["visitor_code"]).astype(float),
        }
    )
    tg = pd.concat([home, vis], ignore_index=True).dropna(subset=["team_code"]).copy()
    tg = tg.sort_values(["team_code", "gameday"])
    tg["rolling_win_10"] = tg.groupby("team_code")["win"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )

    feat = tg.groupby(["season", "team_code"], as_index=False).agg(
        overall_winpct=("win", "mean"),
        last10_winpct=("rolling_win_10", "last"),
    )

    opp_wp = feat[["season", "team_code", "overall_winpct"]].rename(
        columns={"team_code": "opp_code", "overall_winpct": "opp_wp"}
    )
    sos = (
        tg.merge(opp_wp, on=["season", "opp_code"], how="left")
        .groupby(["season", "team_code"], as_index=False)["opp_wp"]
        .mean()
        .rename(columns={"opp_wp": "sos_proxy"})
    )
    feat = feat.merge(sos, on=["season", "team_code"], how="left")

    conf = (
        tg.groupby(["season", "team_code", "conference_proxy"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .drop_duplicates(["season", "team_code"])
        .drop(columns=["size"])
    )
    feat = feat.merge(conf, on=["season", "team_code"], how="left")

    elo = pd.read_parquet(elo_file).copy().rename(columns={"code": "team_code", "elo": "end_elo"})
    elo = elo[elo["season"] == season][["season", "team_code", "end_elo"]]
    feat = feat.merge(elo, on=["season", "team_code"], how="left")

    for col in ["overall_winpct", "last10_winpct", "sos_proxy", "end_elo"]:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")

    feat["league"] = league_name
    return feat


def enrich_team_bracket_features(selection_df: pd.DataFrame, team_stats_file: Path) -> pd.DataFrame:
    ts = pd.read_parquet(team_stats_file).copy()
    if "season" in ts.columns:
        ts = ts[ts["season"] == selection_df["season"].iloc[0]].copy()

    num_cols = ["pts", "fga", "fta", "oreb", "to", "g"]
    for c in num_cols:
        if c in ts.columns:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

    poss = ts["fga"] + 0.44 * ts["fta"] - ts["oreb"] + ts["to"]
    ts["o_ppp"] = ts["pts"] / poss.replace(0, np.nan)
    ts["pace"] = poss / ts["g"].replace(0, np.nan)

    for c in ["o_ppp", "pace", "winpct"]:
        if c not in ts.columns:
            ts[c] = np.nan

    out = selection_df.merge(ts[["team_code", "o_ppp", "pace", "winpct"]], on="team_code", how="left")
    for c in ["winpct", "o_ppp", "pace", "end_elo", "overall_winpct"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["overall_winpct"] = out["overall_winpct"].fillna(out["winpct"])
    return out


def predict_selection_field(selection_feat: pd.DataFrame, model, team_lookup: pd.DataFrame, n_select: int = 68) -> pd.DataFrame:
    X = selection_feat[
        ["overall_winpct", "end_elo", "sos_proxy", "last10_winpct", "conference_proxy", "league"]
    ].copy()
    probs = model.predict_proba(X)[:, 1]
    out = selection_feat.copy()
    out["selection_prob"] = probs
    out = out.sort_values(["selection_prob", "end_elo"], ascending=[False, False]).reset_index(drop=True)
    out["pred_selected"] = 0
    if len(out):
        out.loc[: min(n_select, len(out)) - 1, "pred_selected"] = 1

    out = out.merge(team_lookup, on="team_code", how="left")
    out["team_name"] = out["team_name"].fillna(out["team_code"].astype(str))
    return out


def _to_float(val) -> float:
    v = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.isna(v):
        return 0.0
    return float(v)


def matchup_prob(a: pd.Series, b: pd.Series, model, league_name: str) -> float:
    row = pd.DataFrame(
        [
            {
                "elo_diff": _to_float(a.get("end_elo")) - _to_float(b.get("end_elo")),
                "winpct_diff": _to_float(a.get("overall_winpct")) - _to_float(b.get("overall_winpct")),
                "ppp_diff": _to_float(a.get("o_ppp")) - _to_float(b.get("o_ppp")),
                "pace_diff": _to_float(a.get("pace")) - _to_float(b.get("pace")),
                "home_favored": 1 if (_to_float(a.get("end_elo")) - _to_float(b.get("end_elo"))) > 0 else 0,
                "neutral": "Y",
                "league": league_name,
            }
        ]
    )
    return float(model.predict_proba(row)[:, 1][0])


def _run_round(participants: pd.DataFrame, round_name: str, model, league_name: str) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    if len(participants) <= 1:
        return participants.iloc[0:0], []

    ordered = participants.sort_values("seed_rank").reset_index(drop=True)
    n = len(ordered) - (len(ordered) % 2)
    if n <= 1:
        return ordered.iloc[0:0], []

    i = 0
    j = n - 1
    slot = 0
    records: List[Dict[str, object]] = []
    winners: List[Dict[str, object]] = []

    while i < j:
        a = ordered.iloc[i]
        b = ordered.iloc[j]
        p = matchup_prob(a, b, model, league_name)
        winner = a if p >= 0.5 else b
        rec = {
            "round": round_name,
            "seed_a": int(a["seed_rank"]),
            "team_a": a["team_name"],
            "seed_b": int(b["seed_rank"]),
            "team_b": b["team_name"],
            "p_team_a_win": float(p),
            "winner": winner["team_name"],
            "winner_seed": int(winner["seed_rank"]),
            "slot_a": slot,
            "slot_b": slot + 1,
            "team_a_elo": _to_float(a.get("end_elo")),
            "team_b_elo": _to_float(b.get("end_elo")),
        }
        records.append(rec)
        winners.append(winner.to_dict())
        slot += 2
        i += 1
        j -= 1

    next_df = pd.DataFrame(winners).sort_values("seed_rank").reset_index(drop=True)
    return next_df, records


def project_bracket(selection_df: pd.DataFrame, model, league_name: str):
    field = selection_df[selection_df["pred_selected"] == 1].copy().sort_values(
        ["selection_prob", "end_elo"], ascending=[False, False]
    ).reset_index(drop=True)
    field["seed_rank"] = np.arange(1, len(field) + 1)

    rounds: List[Dict[str, object]] = []
    bracket_rows: List[Dict[str, object]] = []

    playin = field[field["seed_rank"] >= 61].sort_values("seed_rank").reset_index(drop=True)
    main = field[field["seed_rank"] <= 60].sort_values("seed_rank").reset_index(drop=True)
    by_seed = {int(r.seed_rank): r._asdict() for r in playin.itertuples()} if len(playin) else {}

    play_records: List[Dict[str, object]] = []
    playin_winners = []
    for pair_idx, (seed_a, seed_b) in enumerate([(61, 68), (62, 67), (63, 66), (64, 65)], start=0):
        if seed_a not in by_seed or seed_b not in by_seed:
            continue
        a = pd.Series(by_seed[seed_a])
        b = pd.Series(by_seed[seed_b])
        p = matchup_prob(a, b, model, league_name)
        winner = a if p >= 0.5 else b
        play_records.append(
            {
                "round": "First Four",
                "seed_a": int(seed_a),
                "team_a": a["team_name"],
                "seed_b": int(seed_b),
                "team_b": b["team_name"],
                "p_team_a_win": float(p),
                "winner": winner["team_name"],
                "winner_seed": int(winner["seed_rank"]),
                "slot_a": pair_idx * 2,
                "slot_b": pair_idx * 2 + 1,
                "team_a_elo": _to_float(a.get("end_elo")),
                "team_b_elo": _to_float(b.get("end_elo")),
            }
        )
        playin_winners.append(winner.to_dict())
    rounds.append({"round": "First Four", "participants": playin, "games": play_records})
    bracket_rows.extend(play_records)

    if playin_winners:
        playin_field = pd.DataFrame(playin_winners).sort_values("seed_rank").reset_index(drop=True)
        playin_field["seed_rank"] = np.arange(61, 61 + len(playin_field))
        current = pd.concat([main, playin_field], ignore_index=True).sort_values("seed_rank").reset_index(drop=True)
    else:
        current = main.copy()

    for rn in ROUND_ORDER[1:]:
        if len(current) <= 1:
            break
        next_participants, recs = _run_round(current, rn, model, league_name)
        rounds.append({"round": rn, "participants": current.copy(), "games": recs})
        bracket_rows.extend(recs)
        current = next_participants.reset_index(drop=True)

    champion = None
    if len(current) == 1:
        champion = current.iloc[0]["team_name"]
    elif rounds:
        champ_game = next((r for r in reversed(rounds) if r["games"]), None)
        if champ_game:
            champion = champ_game["games"][-1]["winner"]

    return field, pd.DataFrame(bracket_rows), champion, rounds


def _shorten(name: str, max_chars: int = 26) -> str:
    text = (name or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1]}…"


def _fmt_pct(v: float) -> str:
    return "N/A" if pd.isna(v) else f"{float(v) * 100:0.1f}%"


REGION_NAMES = {
    "MBB": ["South", "West", "East", "Midwest"],
    "WBB": ["Albany 1", "Spokane 1", "Birmingham 2", "Albany 2"],
}

REGION_PAIRINGS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGIONAL_ROUNDS = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]


def _seed_group_order(seed_line: int) -> List[int]:
    return [0, 1, 2, 3] if seed_line % 2 == 1 else [3, 2, 1, 0]


def _center_positions(y0: float, height: float, count: int) -> List[float]:
    return [y0 + height - ((i + 0.5) * height / count) for i in range(count)]


def _winner_from_game(team_a: pd.Series, team_b: pd.Series, model, league_name: str) -> Dict[str, object]:
    p_team_a = matchup_prob(team_a, team_b, model, league_name)
    winner = team_a if p_team_a >= 0.5 else team_b
    return {
        "team_a": team_a.to_dict(),
        "team_b": team_b.to_dict(),
        "winner": winner.to_dict(),
        "p_team_a_win": float(p_team_a),
    }


def build_official_bracket_structure(field: pd.DataFrame, model, league_name: str) -> Dict[str, object]:
    selected = (
        field[field["pred_selected"] == 1]
        .sort_values(["selection_prob", "end_elo"], ascending=[False, False])
        .reset_index(drop=True)
        .iloc[:68]
        .copy()
    )
    if len(selected) < 68:
        raise ValueError(f"Expected 68 selected teams for {league_name}, found {len(selected)}")

    region_names = REGION_NAMES[league_name]
    direct = selected.iloc[:60].copy().reset_index(drop=True)
    playin_pool = selected.iloc[60:68].copy().reset_index(drop=True)

    regions: List[Dict[int, Dict[str, object]]] = [{ } for _ in region_names]
    for seed_line in range(1, 16):
        chunk = direct.iloc[(seed_line - 1) * 4 : seed_line * 4].to_dict("records")
        for region_idx, team in zip(_seed_group_order(seed_line), chunk):
            team["seed"] = seed_line
            team["region"] = region_names[region_idx]
            regions[region_idx][seed_line] = team

    playin_games = []
    for region_idx, (a_idx, b_idx) in enumerate([(0, 7), (1, 6), (2, 5), (3, 4)]):
        team_a = pd.Series(playin_pool.iloc[a_idx].to_dict())
        team_b = pd.Series(playin_pool.iloc[b_idx].to_dict())
        team_a["seed"] = 16
        team_b["seed"] = 16
        team_a["region"] = region_names[region_idx]
        team_b["region"] = region_names[region_idx]
        result = _winner_from_game(team_a, team_b, model, league_name)
        result["region"] = region_names[region_idx]
        result["seed"] = 16
        playin_games.append(result)
        winner = result["winner"]
        winner["seed"] = 16
        winner["region"] = region_names[region_idx]
        regions[region_idx][16] = winner

    region_results = []
    region_champions = []
    for region_idx, region_name in enumerate(region_names):
        seed_map = regions[region_idx]
        games_by_round: Dict[str, List[Dict[str, object]]] = {}

        opening_round = []
        current_round = []
        for seed_a, seed_b in REGION_PAIRINGS:
            result = _winner_from_game(pd.Series(seed_map[seed_a]), pd.Series(seed_map[seed_b]), model, league_name)
            opening_round.append(result)
            current_round.append(result["winner"])
        games_by_round["Round of 64"] = opening_round

        for round_name in REGIONAL_ROUNDS[1:]:
            next_games = []
            next_round = []
            for idx in range(0, len(current_round), 2):
                result = _winner_from_game(pd.Series(current_round[idx]), pd.Series(current_round[idx + 1]), model, league_name)
                next_games.append(result)
                next_round.append(result["winner"])
            games_by_round[round_name] = next_games
            current_round = next_round

        champion = current_round[0]
        champion["region"] = region_name
        region_results.append({"name": region_name, "games": games_by_round, "champion": champion})
        region_champions.append(champion)

    semifinal_left = _winner_from_game(pd.Series(region_champions[0]), pd.Series(region_champions[1]), model, league_name)
    semifinal_right = _winner_from_game(pd.Series(region_champions[2]), pd.Series(region_champions[3]), model, league_name)
    championship = _winner_from_game(
        pd.Series(semifinal_left["winner"]),
        pd.Series(semifinal_right["winner"]),
        model,
        league_name,
    )

    return {
        "league": league_name,
        "regions": region_results,
        "first_four": playin_games,
        "semifinals": [semifinal_left, semifinal_right],
        "championship": championship,
        "champion": championship["winner"],
    }


def _draw_team_box(ax: plt.Axes, x: float, y: float, width: float, height: float, game: Dict[str, object], side: str) -> None:
    team_a = game["team_a"]
    team_b = game["team_b"]
    winner_name = game["winner"]["team_name"]
    top_is_winner = team_a["team_name"] == winner_name
    bottom_is_winner = team_b["team_name"] == winner_name

    ax.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.002",
            facecolor="#ffffff",
            edgecolor="#969eaf",
            linewidth=0.8,
            zorder=3,
        )
    )
    ax.plot([x, x + width], [y, y], color="#d8dde7", lw=0.7, zorder=4)

    pad = 0.0045
    ha = "left"
    text_x = x + pad
    if side == "right":
        ha = "right"
        text_x = x + width - pad

    ax.text(
        text_x,
        y + height * 0.22,
        f"{int(team_a['seed'])} {_shorten(str(team_a['team_name']), 22)}",
        ha=ha,
        va="center",
        fontsize=6.2,
        color="#16253f",
        fontweight="bold" if top_is_winner else "normal",
        zorder=5,
    )
    ax.text(
        text_x,
        y - height * 0.22,
        f"{int(team_b['seed'])} {_shorten(str(team_b['team_name']), 22)}",
        ha=ha,
        va="center",
        fontsize=6.2,
        color="#16253f",
        fontweight="bold" if bottom_is_winner else "normal",
        zorder=5,
    )


def _draw_region(ax: plt.Axes, region: Dict[str, object], x_cols: List[float], y0: float, height: float, side: str, semifinal_anchor: Tuple[float, float]) -> None:
    box_w = 0.125
    box_h = 0.034
    centers: Dict[str, List[float]] = {"Round of 64": _center_positions(y0, height, 8)}
    for round_name, prev_name in zip(REGIONAL_ROUNDS[1:], REGIONAL_ROUNDS[:-1]):
        prev = centers[prev_name]
        centers[round_name] = [(prev[idx] + prev[idx + 1]) / 2.0 for idx in range(0, len(prev), 2)]

    for round_name, x in zip(REGIONAL_ROUNDS, x_cols):
        games = region["games"][round_name]
        ys = centers[round_name]
        next_round = None
        next_x = None
        if round_name != "Elite 8":
            next_round = REGIONAL_ROUNDS[REGIONAL_ROUNDS.index(round_name) + 1]
            next_x = x_cols[REGIONAL_ROUNDS.index(round_name) + 1]

        for idx, (game, y) in enumerate(zip(games, ys)):
            _draw_team_box(ax, x, y, box_w, box_h, game, side)
            if next_round is None:
                y_next = semifinal_anchor[1]
                x_next = semifinal_anchor[0]
            else:
                y_next = centers[next_round][idx // 2]
                x_next = next_x

            if side == "left":
                x_from = x + box_w
                x_mid = x_from + (x_next - x_from) * 0.54
                x_to = x_next
            else:
                x_from = x
                x_mid = x_from - ((x_from - (x_next + box_w)) * 0.54)
                x_to = x_next + box_w

            y_from = y + box_h * 0.22 if game["winner"]["team_name"] == game["team_a"]["team_name"] else y - box_h * 0.22
            ax.plot([x_from, x_mid], [y_from, y_from], color="#7f8798", lw=1.15, zorder=2)
            ax.plot([x_mid, x_mid], [y_from, y_next], color="#7f8798", lw=1.15, zorder=2)
            ax.plot([x_mid, x_to], [y_next, y_next], color="#7f8798", lw=1.15, zorder=2)

    label_y = y0 + height + 0.016
    label_x = x_cols[0] + (box_w / 2)
    if side == "right":
        label_x = x_cols[0] + (box_w / 2)
    ax.text(label_x, label_y, region["name"].upper(), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#17346b")


def _draw_center_game(ax: plt.Axes, x: float, y: float, width: float, height: float, game: Dict[str, object], title: str) -> None:
    ax.text(x + width / 2, y + height / 2 + 0.018, title, ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#17346b")
    _draw_team_box(ax, x, y, width, height, game, "left")


def _draw_first_four(ax: plt.Axes, games: List[Dict[str, object]]) -> None:
    if not games:
        return
    ax.text(0.5, 0.885, "FIRST FOUR", ha="center", va="center", fontsize=11, fontweight="bold", color="#17346b")
    positions = [(0.17, 0.845), (0.39, 0.845), (0.61, 0.845), (0.83, 0.845)]
    for (x, y), game in zip(positions, games):
        ax.text(x + 0.0625, y + 0.03, game["region"].upper(), ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#5b6781")
        _draw_team_box(ax, x, y, 0.125, 0.03, game, "left")


def _render_official_bracket(ax: plt.Axes, structure: Dict[str, object], title: str, as_of: pd.Timestamp) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#f7f7f5")

    ax.add_patch(FancyBboxPatch((0.015, 0.942), 0.97, 0.045, boxstyle="round,pad=0.004", facecolor="#17346b", edgecolor="#17346b"))
    ax.text(
        0.5,
        0.964,
        f"{title.upper()}  |  AS OF {pd.to_datetime(as_of).strftime('%Y-%m-%d')}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )

    ax.text(0.03, 0.965, "NCAA", ha="left", va="center", fontsize=10, fontweight="bold", color="#dbe6ff")
    ax.text(0.97, 0.965, "PRINTABLE FORECAST", ha="right", va="center", fontsize=8.5, fontweight="bold", color="#dbe6ff")

    _draw_first_four(ax, structure["first_four"])

    round_headers_left = [0.095, 0.23, 0.365, 0.485]
    round_headers_right = [0.905, 0.77, 0.635, 0.515]
    for x, label in zip(round_headers_left, REGIONAL_ROUNDS):
        ax.text(x, 0.79, label.upper(), ha="center", va="bottom", fontsize=6.8, color="#5a657a", fontweight="bold")
    for x, label in zip(round_headers_right, REGIONAL_ROUNDS):
        ax.text(x, 0.79, label.upper(), ha="center", va="bottom", fontsize=6.8, color="#5a657a", fontweight="bold")

    semifinal_left_anchor = (0.445, 0.565)
    semifinal_right_anchor = (0.555, 0.565)

    _draw_region(ax, structure["regions"][0], [0.02, 0.155, 0.29, 0.39], 0.515, 0.255, "left", semifinal_left_anchor)
    _draw_region(ax, structure["regions"][1], [0.02, 0.155, 0.29, 0.39], 0.11, 0.255, "left", (0.445, 0.275))
    _draw_region(ax, structure["regions"][2], [0.855, 0.72, 0.585, 0.485], 0.515, 0.255, "right", semifinal_right_anchor)
    _draw_region(ax, structure["regions"][3], [0.855, 0.72, 0.585, 0.485], 0.11, 0.255, "right", (0.555, 0.275))

    ax.add_patch(FancyBboxPatch((0.418, 0.325), 0.164, 0.33, boxstyle="round,pad=0.006", facecolor="#f1f2ee", edgecolor="#d5d9e2", linewidth=1.0, zorder=1))
    ax.text(0.5, 0.635, "FINAL FOUR", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#17346b")
    _draw_center_game(ax, 0.395, 0.565, 0.115, 0.038, structure["semifinals"][0], structure["regions"][0]["name"].upper())
    _draw_center_game(ax, 0.49, 0.565, 0.115, 0.038, structure["semifinals"][1], structure["regions"][2]["name"].upper())
    _draw_center_game(ax, 0.4425, 0.405, 0.115, 0.042, structure["championship"], "CHAMPIONSHIP")

    left_semi_winner = structure["semifinals"][0]["winner"]["team_name"]
    right_semi_winner = structure["semifinals"][1]["winner"]["team_name"]
    title_y_left = 0.565 + (0.038 * 0.22 if structure["semifinals"][0]["team_a"]["team_name"] == left_semi_winner else -0.038 * 0.22)
    title_y_right = 0.565 + (0.038 * 0.22 if structure["semifinals"][1]["team_a"]["team_name"] == right_semi_winner else -0.038 * 0.22)
    champ_y = 0.405 + (0.042 * 0.22 if structure["championship"]["team_a"]["team_name"] == structure["champion"]["team_name"] else -0.042 * 0.22)

    ax.plot([0.51, 0.51], [title_y_left, champ_y], color="#7f8798", lw=1.2, zorder=2)
    ax.plot([0.49, 0.51], [title_y_left, title_y_left], color="#7f8798", lw=1.2, zorder=2)
    ax.plot([0.605, 0.51], [title_y_right, title_y_right], color="#7f8798", lw=1.2, zorder=2)

    ax.text(0.5, 0.352, str(structure["champion"]["team_name"]), ha="center", va="center", fontsize=12.2, fontweight="bold", color="#17346b")
    ax.text(0.5, 0.329, "Projected Champion", ha="center", va="center", fontsize=7.2, color="#52617f")


def draw_official_bracket(structure: Dict[str, object], as_of: pd.Timestamp, output_path: Path, title: str) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(18, 11), dpi=220)
    fig.patch.set_facecolor("#e9eaee")
    _render_official_bracket(ax, structure, title, as_of)
    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_combo_bracket(mbb_structure: Dict[str, object], wbb_structure: Dict[str, object], as_of: pd.Timestamp, output_path: Path) -> None:
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(18, 20), dpi=180)
    fig.patch.set_facecolor("#e9eaee")
    _render_official_bracket(axes[0], mbb_structure, "Men's NCAA Tournament Projection", as_of)
    _render_official_bracket(axes[1], wbb_structure, "Women's NCAA Tournament Projection", as_of)
    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_pipeline() -> None:
    sel_model = joblib.load(OUT / "selection_pooled_model.joblib")
    brk_mbb_model = joblib.load(OUT / "bracket_mbb_model.joblib")
    brk_wbb_model = joblib.load(OUT / "bracket_wbb_model.joblib")

    season_mbb = int(pd.read_parquet(ROOT / "games.parquet")["season"].max())
    season_wbb = int(pd.read_parquet(ROOT / "wbb_games.parquet")["season"].max())
    season = max(season_mbb, season_wbb)

    as_of_mbb = _latest_gameday_for_season(ROOT / "games.parquet", season)
    as_of_wbb = _latest_gameday_for_season(ROOT / "wbb_games.parquet", season)
    as_of = max(as_of_mbb, as_of_wbb)

    mbb_lookup = _load_team_lookup(ROOT / "teams.parquet", ROOT / "team_stats.parquet", season)
    wbb_lookup = _load_team_lookup(ROOT / "wbb_teams.parquet", ROOT / "wbb_team_stats.parquet", season)

    mbb_sel = build_selection_features_live(ROOT / "games.parquet", ROOT / "elo.parquet", season, "MBB")
    wbb_sel = build_selection_features_live(ROOT / "wbb_games.parquet", ROOT / "wbb_elo.parquet", season, "WBB")

    mbb_sel = enrich_team_bracket_features(mbb_sel, ROOT / "team_stats.parquet")
    wbb_sel = enrich_team_bracket_features(wbb_sel, ROOT / "wbb_team_stats.parquet")

    mbb_pred = predict_selection_field(mbb_sel, sel_model, mbb_lookup, n_select=68)
    wbb_pred = predict_selection_field(wbb_sel, sel_model, wbb_lookup, n_select=68)

    mbb_pred.to_csv(OUT / f"live_{season}_selection_mbb.csv", index=False)
    wbb_pred.to_csv(OUT / f"live_{season}_selection_wbb.csv", index=False)

    mbb_field, mbb_bracket, mbb_champ, mbb_rounds = project_bracket(mbb_pred, brk_mbb_model, "MBB")
    wbb_field, wbb_bracket, wbb_champ, wbb_rounds = project_bracket(wbb_pred, brk_wbb_model, "WBB")
    mbb_official = build_official_bracket_structure(mbb_field, brk_mbb_model, "MBB")
    wbb_official = build_official_bracket_structure(wbb_field, brk_wbb_model, "WBB")

    mbb_field.to_csv(OUT / f"live_{season}_field_mbb.csv", index=False)
    wbb_field.to_csv(OUT / f"live_{season}_field_wbb.csv", index=False)
    mbb_bracket.to_csv(OUT / f"live_{season}_bracket_projection_mbb.csv", index=False)
    wbb_bracket.to_csv(OUT / f"live_{season}_bracket_projection_wbb.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "league": "MBB",
                "selected_teams": int(mbb_pred["pred_selected"].sum()),
                "projected_champion": mbb_official["champion"]["team_name"],
            },
            {
                "league": "WBB",
                "selected_teams": int(wbb_pred["pred_selected"].sum()),
                "projected_champion": wbb_official["champion"]["team_name"],
            },
        ]
    )
    summary.to_csv(OUT / f"live_{season}_forecast_summary.csv", index=False)

    # Remove older outputs before rewriting "best version" images.
    for old in [
        OUT / "ncaa_bracket_official_mbb.png",
        OUT / "ncaa_bracket_official_wbb.png",
        OUT / "ncaa_bracket_official_combo.png",
        OUT / "ncaa_bracket_projection_mbb_latest.png",
        OUT / "ncaa_bracket_projection_wbb_latest.png",
        OUT / "ncaa_bracket_projection_mbb_wbb_latest.png",
    ]:
        if old.exists():
            old.unlink()

    draw_official_bracket(
        mbb_official,
        as_of,
        OUT / "ncaa_bracket_official_mbb.png",
        title=f"{season} MEN'S NCAA TOURNAMENT PROJECTION",
    )
    draw_official_bracket(
        wbb_official,
        as_of,
        OUT / "ncaa_bracket_official_wbb.png",
        title=f"{season} WOMEN'S NCAA TOURNAMENT PROJECTION",
    )
    draw_combo_bracket(mbb_official, wbb_official, as_of, OUT / "ncaa_bracket_official_combo.png")

    print(f"Saved season={season} forecasts (as of {as_of.date()})")
    print(OUT / f"live_{season}_selection_mbb.csv")
    print(OUT / f"live_{season}_selection_wbb.csv")
    print(OUT / f"live_{season}_field_mbb.csv")
    print(OUT / f"live_{season}_field_wbb.csv")
    print(OUT / f"live_{season}_bracket_projection_mbb.csv")
    print(OUT / f"live_{season}_bracket_projection_wbb.csv")
    print(OUT / f"live_{season}_forecast_summary.csv")
    print(OUT / "ncaa_bracket_official_mbb.png")
    print(OUT / "ncaa_bracket_official_wbb.png")
    print(OUT / "ncaa_bracket_official_combo.png")


if __name__ == "__main__":
    run_pipeline()
