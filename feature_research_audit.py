#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


EPS = 1e-6


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return out


def add_team_rate_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    ts = team_stats.copy()
    for c in ts.columns:
        if c not in {"team_code"}:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

    # Possession / per-game fallbacks
    if {"fga", "fta", "to", "oreb"}.issubset(ts.columns):
        ts["poss_est"] = ts["fga"] + 0.44 * ts["fta"] + ts["to"] - ts["oreb"]
    else:
        ts["poss_est"] = np.nan

    if "o_ppp" not in ts.columns:
        ts["o_ppp"] = np.nan
    if {"pts", "poss_est"}.issubset(ts.columns):
        ts["o_ppp"] = ts["o_ppp"].fillna(np.where(ts["poss_est"] > 0, ts["pts"] / (ts["poss_est"] + EPS), np.nan))

    if "pace" not in ts.columns:
        ts["pace"] = np.nan
    if {"poss_est", "g"}.issubset(ts.columns):
        ts["pace"] = ts["pace"].fillna(np.where(ts["g"] > 0, ts["poss_est"] / (ts["g"] + EPS), np.nan))

    # Shooting / ball security rates
    if {"fgm", "fga"}.issubset(ts.columns):
        ts["fg_pct"] = np.where(ts["fga"] > 0, ts["fgm"] / (ts["fga"] + EPS), np.nan)
    if {"ftm", "fta"}.issubset(ts.columns):
        ts["ft_pct"] = np.where(ts["fta"] > 0, ts["ftm"] / (ts["fta"] + EPS), np.nan)
    if {"fta", "fga"}.issubset(ts.columns):
        ts["ft_rate"] = np.where(ts["fga"] > 0, ts["fta"] / (ts["fga"] + EPS), np.nan)
    if {"to", "poss_est"}.issubset(ts.columns):
        ts["to_rate"] = np.where(ts["poss_est"] > 0, ts["to"] / (ts["poss_est"] + EPS), np.nan)
    if {"oreb", "reb"}.issubset(ts.columns):
        ts["oreb_share"] = np.where(ts["reb"] > 0, ts["oreb"] / (ts["reb"] + EPS), np.nan)

    return ts


@dataclass
class LeagueConfig:
    name: str
    games_file: str
    team_stats_file: str
    elo_file: str


def load_model_base(root: Path, cfg: LeagueConfig) -> pd.DataFrame:
    games = clean_columns(pd.read_parquet(root / cfg.games_file))
    team_stats = clean_columns(pd.read_parquet(root / cfg.team_stats_file))
    elo = clean_columns(pd.read_parquet(root / cfg.elo_file))

    for df in (games, team_stats, elo):
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    ts_keep = [
        "team_code",
        "season",
        "pts",
        "g",
        "min",
        "reb",
        "oreb",
        "dreb",
        "to",
        "fgm",
        "fga",
        "ftm",
        "fta",
        "winpct",
        "o_ppp",
        "pace",
    ]
    ts_keep = [c for c in ts_keep if c in team_stats.columns]
    ts = add_team_rate_features(team_stats[ts_keep].copy())

    elo_keep = [c for c in ["code", "elo", "season"] if c in elo.columns]
    e = elo[elo_keep].copy()
    if "elo" in e.columns:
        e["elo"] = pd.to_numeric(e["elo"], errors="coerce")

    m = games.copy()
    if "season" not in m.columns:
        raise ValueError(f"{cfg.name}: games missing season")

    # Merge team stats by team+season
    home_ts = ts.rename(columns={"team_code": "home_code"})
    vis_ts = ts.rename(columns={"team_code": "visitor_code"})

    m = m.merge(home_ts, on=[c for c in ["home_code", "season"] if c in home_ts.columns], how="left")
    for c in ts.columns:
        if c not in {"team_code", "season"} and c in m.columns:
            m.rename(columns={c: f"{c}_home"}, inplace=True)

    m = m.merge(vis_ts, on=[c for c in ["visitor_code", "season"] if c in vis_ts.columns], how="left")
    for c in ts.columns:
        if c not in {"team_code", "season"} and c in m.columns and f"{c}_home" in m.columns:
            m.rename(columns={c: f"{c}_vis"}, inplace=True)

    # Merge Elo by team+season
    he = e.rename(columns={"code": "home_code"})
    ve = e.rename(columns={"code": "visitor_code"})

    m = m.merge(he, on=[c for c in ["home_code", "season"] if c in he.columns], how="left")
    if "elo" in m.columns:
        m.rename(columns={"elo": "elo_home"}, inplace=True)

    m = m.merge(ve, on=[c for c in ["visitor_code", "season"] if c in ve.columns], how="left")
    if "elo" in m.columns:
        m.rename(columns={"elo": "elo_vis"}, inplace=True)

    # Final games only + de-dup
    if "gamestatus" in m.columns:
        gs = m["gamestatus"].astype(str).str.strip().str.lower()
        m = m[gs.str.startswith("final")].copy()
    if "id" in m.columns:
        m = m.drop_duplicates(subset=["id"]).copy()

    m["home_win"] = (m["winner_code"] == m["home_code"]).astype(int)

    if "gameday" in m.columns:
        dt = pd.to_datetime(m["gameday"], errors="coerce")
        m["game_month"] = dt.dt.month
        m["game_dow"] = dt.dt.dayofweek
        m["is_postseason_month"] = dt.dt.month.isin([3, 4]).astype("Int64")

    return m


def add_game_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    base_metrics = [
        "elo",
        "o_ppp",
        "pace",
        "winpct",
        "fg_pct",
        "ft_pct",
        "ft_rate",
        "to_rate",
        "oreb_share",
        "poss_est",
    ]

    feature_cols: list[str] = []

    for m in base_metrics:
        h = f"{m}_home"
        v = f"{m}_vis"
        if h in out.columns and v in out.columns:
            diff = f"{m}_diff"
            adiff = f"abs_{m}_diff"
            ratio = f"{m}_ratio"
            summ = f"{m}_sum"
            out[diff] = out[h] - out[v]
            out[adiff] = out[diff].abs()
            out[ratio] = out[h] / (out[v] + EPS)
            out[summ] = out[h] + out[v]
            feature_cols.extend([diff, adiff, ratio, summ])

    # Interaction / nonlinear terms on strong anchors if present
    for a, b in [
        ("elo_diff", "winpct_diff"),
        ("elo_diff", "o_ppp_diff"),
        ("winpct_diff", "o_ppp_diff"),
        ("pace_diff", "o_ppp_diff"),
    ]:
        if a in out.columns and b in out.columns:
            name = f"int_{a}_x_{b}"
            out[name] = out[a] * out[b]
            feature_cols.append(name)

    for c in ["elo_diff", "o_ppp_diff", "pace_diff", "winpct_diff"]:
        if c in out.columns:
            sq = f"sq_{c}"
            out[sq] = out[c] ** 2
            feature_cols.append(sq)

    if "score_home" in out.columns and "score_vis" in out.columns:
        sh = pd.to_numeric(out["score_home"], errors="coerce")
        sv = pd.to_numeric(out["score_vis"], errors="coerce")
        out["game_total"] = sh + sv
        out["margin"] = sh - sv
        out["margin_abs"] = (sh - sv).abs()

    # Include robust calendar features
    for c in ["game_month", "game_dow", "is_postseason_month"]:
        if c in out.columns:
            feature_cols.append(c)

    # De-duplicate feature list while preserving order
    seen = set()
    ordered = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return out, ordered


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "season" not in df.columns or df["season"].dropna().nunique() < 2:
        test = df.sample(frac=0.2, random_state=42)
        train = df.drop(test.index)
        return train, test

    seasons = sorted([int(s) for s in df["season"].dropna().unique()])
    cutoff = seasons[-2] if len(seasons) >= 3 else seasons[-1]
    train = df[df["season"] < cutoff].copy()
    test = df[df["season"] >= cutoff].copy()

    if len(train) < 1000 or len(test) < 500:
        test = df.sample(frac=0.2, random_state=42)
        train = df.drop(test.index)
    return train, test


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 200 or d["y"].nunique() < 2:
        return np.nan
    try:
        a = roc_auc_score(d["y"], d["x"])
        # Make direction agnostic.
        return max(a, 1 - a)
    except Exception:
        return np.nan


def audit_feature_usefulness(df: pd.DataFrame, features: Iterable[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    features = [f for f in features if f in df.columns]
    d = df[features + ["home_win", "season"]].copy()

    # coverage stats
    coverage = pd.Series({f: float(d[f].notna().mean()) for f in features}, name="coverage")
    uniq = pd.Series({f: int(d[f].nunique(dropna=True)) for f in features}, name="nunique")

    # univariate usefulness
    uni = pd.Series({f: univariate_auc(d[f], d["home_win"]) for f in features}, name="univariate_auc")

    # multivariate model + permutation importance
    use_feats = [f for f in features if coverage[f] >= 0.2 and uniq[f] > 1]
    model_df = d[use_feats + ["home_win", "season"]].dropna(subset=["home_win"]).copy()
    train, test = temporal_split(model_df)

    X_train = train[use_feats]
    y_train = train["home_win"]
    X_test = test[use_feats]
    y_test = test["home_win"]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "test_auc": float(roc_auc_score(y_test, p)),
        "n_train": float(len(train)),
        "n_test": float(len(test)),
        "n_features": float(len(use_feats)),
    }

    perm = permutation_importance(pipe, X_test, y_test, n_repeats=4, random_state=42, scoring="roc_auc")
    perm_s = pd.Series(perm.importances_mean, index=use_feats, name="perm_importance")

    out = pd.concat([coverage, uniq, uni, perm_s], axis=1).reset_index().rename(columns={"index": "feature"})
    out["perm_importance"] = out["perm_importance"].fillna(0.0)
    out = out.sort_values(["perm_importance", "univariate_auc", "coverage"], ascending=[False, False, False])

    return out, metrics


def main() -> None:
    root = Path.cwd()

    men_cfg = LeagueConfig("men", "games.parquet", "team_stats.parquet", "elo.parquet")
    women_cfg = LeagueConfig("women", "wbb_games.parquet", "wbb_team_stats.parquet", "wbb_elo.parquet")

    men = load_model_base(root, men_cfg)
    women = load_model_base(root, women_cfg)

    men, men_features = add_game_features(men)
    women, women_features = add_game_features(women)

    # Shared set for pooled analysis
    shared = sorted(set(men_features).intersection(women_features))
    men["is_women"] = 0
    women["is_women"] = 1
    pooled = pd.concat([men, women], ignore_index=True)

    men_report, men_metrics = audit_feature_usefulness(men, men_features)
    women_report, women_metrics = audit_feature_usefulness(women, women_features)
    pooled_report, pooled_metrics = audit_feature_usefulness(pooled, shared + ["is_women"])

    men_path = root / "men_feature_audit.csv"
    women_path = root / "women_feature_audit.csv"
    pooled_path = root / "pooled_feature_audit.csv"

    men_report.to_csv(men_path, index=False)
    women_report.to_csv(women_path, index=False)
    pooled_report.to_csv(pooled_path, index=False)

    print("Saved:")
    print("-", men_path)
    print("-", women_path)
    print("-", pooled_path)

    print("\nModel metrics:")
    print("men:", men_metrics)
    print("women:", women_metrics)
    print("pooled:", pooled_metrics)

    print("\nTop 12 men features:")
    print(men_report.head(12).to_string(index=False))
    print("\nTop 12 women features:")
    print(women_report.head(12).to_string(index=False))
    print("\nTop 12 pooled features:")
    print(pooled_report.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
