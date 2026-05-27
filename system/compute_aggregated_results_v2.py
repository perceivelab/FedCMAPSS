import wandb
import argparse
import pandas as pd
from pathlib import Path
import json

ALGORITHM_ORDER = ["FedAvg", "SCAFFOLD", "FedDyn", "FedCross"]
ARCHITECTURE_ORDER = ["LSTM", "AFT", "AttBiGRU", "RNN", "CNN"]

ALGORITHM_MAP = {
    "FedAvg_RUL": "FedAvg",
    "SCAFFOLD_RUL": "SCAFFOLD",
    "FedDyn_RUL": "FedDyn",
    "FedCross_RUL": "FedCross",
}

MODEL_MAP = {
    "LSTM_RUL": "LSTM",
    "AFT_RUL": "AFT",
    "AttBiGRU_RUL": "AttBiGRU",
    "RNN_RUL": "RNN",
    "Chen_CNN_RUL": "CNN",
}

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Aggregate W&B runs for a project, with optional filters on config keys."
    )

    # Output
    parser.add_argument("--out", default=None, help="Optional output CSV path (e.g. runs.csv)")

    # WandB
    parser.add_argument("--entity", default="bomber-team", help="W&B entity/team")
    parser.add_argument("--project", default="Rocket-FL-sigmoid", help="W&B project")

    # Basic run selection
    parser.add_argument("--state", default="finished",
                        help="Filter by run state (finished, running, crashed, etc.)")
    parser.add_argument("--order", default="-created_at",
                        help="Order string for W&B API (default: -created_at)")
    parser.add_argument("--per-page", type=int, default=200,
                        help="Pagination size for W&B API (default: 200)")

    # Experiment filters
    parser.add_argument("--algorithm", type=str,
        choices=["FedAvg_RUL", "SCAFFOLD_RUL", "FedProx_RUL", "FedDyn_RUL", "FedCross_RUL", "Centralized_RUL", "Local_RUL"],
        default=None,
        help="Filter by config.algorithm (if omitted, no algorithm filter is applied)"
    )
    parser.add_argument("--model", type=str,
        choices=["LSTM_RUL", "AFT_RUL", "AttBiGRU_RUL", "RNN_RUL", "Chen_CNN_RUL"],
        default=None,
        help="Filter by config.model"
    )
    parser.add_argument("--task", type=str, default=None, help="Filter by config.task")
    parser.add_argument("--global_rounds", type=int, default=None, help="Filter by config.global_rounds")
    parser.add_argument("--local_epochs", type=int, choices=[1, 5, 10], default=None, help="Filter by config.local_epochs")

    # Boolean flag
    parser.add_argument("--no_clip_rul", action="store_true",
                        help="If set, filter config.no_clip_rul=True")

    # Step-wise extraction from history
    parser.add_argument("--steps", type=int, nargs="*", default=None,
        help="Specific _step values to extract from history (e.g. --steps 20 50 100)")
    parser.add_argument("--missing-configs-json", type=str,
        help="Optional JSON file path to backfill missing config.* fields based on config.config_id."
    )
    parser.add_argument("--group-by", type=str, choices=["architecture", "algorithm_local_epochs"], default=None,
        help=(
            "Optional grouped aggregation mode. "
            "'architecture' groups by algorithm + model + local_epochs; "
            "'algorithm_local_epochs' groups by algorithm + local_epochs (averaging across architectures)."
        )
    )
    return parser

def parse_filters_from_args(args) -> dict:
    filters = {}

    # run state
    if args.state:
        filters["state"] = args.state

    # config filters
    if args.algorithm is not None:
        filters["config.algorithm"] = args.algorithm
    if args.model is not None:
        filters["config.model"] = args.model
    if args.task is not None:
        filters["config.task"] = args.task
    if args.global_rounds is not None:
        filters["config.global_rounds"] = args.global_rounds
    if args.local_epochs is not None:
        filters["config.local_epochs"] = args.local_epochs

    if args.no_clip_rul:
        filters["config.no_clip_rul"] = True

    return filters

def aggregate_runs(runs):
    rows = []
    for r in runs:
        try:
            config_dict = dict(r.config)
        except Exception as exc:
            config_dict = dict(getattr(r, "_attrs", {}).get("config", {}) or {})
            print(f"[WARN] Could not read run.config for run {getattr(r, 'id', 'unknown')}: {exc}. Using raw attrs fallback.")

        try:
            summary_dict = dict(r.summary)
        except Exception as exc:
            summary_dict = dict(getattr(r, "_attrs", {}).get("summaryMetrics", {}) or {})
            print(f"[WARN] Could not read run.summary for run {getattr(r, 'id', 'unknown')}: {exc}. Using raw attrs fallback.")

        if not isinstance(config_dict, dict):
            config_dict = {}
        if not isinstance(summary_dict, dict):
            summary_dict = {}

        rows.append({
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "created_at": r.created_at,
            "tags": list(r.tags),
            **{f"config.{k}": v for k, v in config_dict.items()},
            **{f"summary.{k}": v for k, v in summary_dict.items()},
        })
    return pd.DataFrame(rows)

def backfill_configs_from_json(df: pd.DataFrame, json_path: str, config_id_base: str = "auto") -> pd.DataFrame:
    """
    Backfills missing config.* fields in df using a JSON file that maps config.config_id to config values.
        - df: DataFrame with runs data, expected to have 'config.config_id' column (int or str)
        - json_path: path to JSON file containing a list of config dicts (without config.config_id)
        - config_id_base: "0", "1", or "auto" to determine if config.config_id in JSON is 0-based or 1-based
        
    Returns a new DataFrame with missing config fields filled where possible.
    """
    if df is None or df.empty:
        return df

    if "config.config_id" not in df.columns:
        print("[INFO] 'config.config_id' column not found in DataFrame. Backfill skipped.")
        return df

    path = Path(json_path)
    if not path.exists():
        print(f"[WARN] File JSON not found: {json_path}. Backfill skipped.")
        return df

    with open(path, "r", encoding="utf-8") as f:
        cfg_list = json.load(f)


    if not isinstance(cfg_list, list) or len(cfg_list) == 0:
        print("[WARN] Empty or invalid JSON format (expected a non-empty list). Backfill skipped.")
        return df

    cfg_df_raw = pd.DataFrame(cfg_list)

    out = df.copy()
    out["config.config_id"] = pd.to_numeric(out["config.config_id"], errors="coerce").astype("Int64")

    candidate_cols = ["config.local_epochs", "config.model", "config.algorithm", "config.task"]
    missing_mask = pd.Series(False, index=out.index)
    for c in candidate_cols:
        if c in out.columns:
            missing_mask = missing_mask | out[c].isna()

    if not missing_mask.any():
        print("[INFO] No missing config fields to backfill.")
        return out

    missing_ids = set(out.loc[missing_mask, "config.config_id"].dropna().astype(int).tolist())

    def build_lookup(base_zero: bool) -> pd.DataFrame:
        cfg_df = cfg_df_raw.copy()
        if base_zero:
            cfg_df.insert(0, "config.config_id", range(len(cfg_df)))
        else:
            cfg_df.insert(0, "config.config_id", range(1, len(cfg_df) + 1))

        rename_map = {
            c: (c if c == "config.config_id" else f"config.{c}")
            for c in cfg_df.columns
        }
        cfg_df = cfg_df.rename(columns=rename_map)
        cfg_df["config.config_id"] = pd.to_numeric(cfg_df["config.config_id"], errors="coerce").astype("Int64")
        return cfg_df

    if config_id_base == "0":
        lookup = build_lookup(base_zero=True)
        chosen_base = 0
    elif config_id_base == "1":
        lookup = build_lookup(base_zero=False)
        chosen_base = 1
    else:
        lookup0 = build_lookup(base_zero=True)
        lookup1 = build_lookup(base_zero=False)

        ids0 = set(lookup0["config.config_id"].dropna().astype(int).tolist())
        ids1 = set(lookup1["config.config_id"].dropna().astype(int).tolist())

        hits0 = len(missing_ids & ids0)
        hits1 = len(missing_ids & ids1)

        if hits1 > hits0:
            lookup = lookup1
            chosen_base = 1
        else:
            lookup = lookup0
            chosen_base = 0

    print(f"[INFO] Backfill config: config_id {chosen_base}-based")

    merged = out.merge(lookup, on="config.config_id", how="left", suffixes=("", "_json"))
    json_cols = [c for c in lookup.columns if c != "config.config_id"]
    filled_counter = {}
    for c in json_cols:
        c_json = f"{c}_json"
        if c_json not in merged.columns:
            continue

        before_missing = merged[c].isna().sum() if c in merged.columns else len(merged)

        if c in merged.columns:
            merged[c] = merged[c].where(merged[c].notna(), merged[c_json])
        else:
            merged[c] = merged[c_json]

        after_missing = merged[c].isna().sum()
        filled_counter[c] = int(before_missing - after_missing)

    drop_cols = [c for c in merged.columns if c.endswith("_json")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    print("[INFO] Backfill completed. Configs filled from JSON:")
    for k, v in filled_counter.items():
        if v > 0:
            print(f"  - {k}: {v}")

    return merged

def compute_grouped_summary_stats(df, group_cols, metric_cols):
    """
    Computes count/mean/std of specified metric_cols grouped by group_cols.

    Returns a DataFrame with one row per group and columns for count/mean/std of each metric.
    """
    if not group_cols:
        return None

    missing_groups = [c for c in group_cols if c not in df.columns]
    if missing_groups:
        print(f"[WARN] Missing group columns in DataFrame: {missing_groups}. Grouped stats cannot be computed.")
        return None

    valid_metrics = [m for m in metric_cols if m in df.columns]
    if not valid_metrics:
        print("[WARN] None of the specified metric columns found in DataFrame. Grouped stats cannot be computed.")
        return None

    work = df.copy()
    for c in group_cols:
        work[c] = work[c].apply(_to_hashable_scalar)

    grouped = (
        work.groupby(group_cols)[valid_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join([str(x) for x in col if x]))
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols

    return grouped

def compute_centralized_model_split_stats(df: pd.DataFrame, metric_cols):
    """
    Centralized case: group by model by averaging over splits first,
    then compute count/mean/std across split-level values.
    """
    required_group_cols = ["config.model"]
    missing_groups = [c for c in required_group_cols if c not in df.columns]
    if missing_groups:
        print(f"[WARN] Missing required columns for centralized aggregation: {missing_groups}")
        return None

    valid_metrics = [m for m in metric_cols if m in df.columns]
    if not valid_metrics:
        print("[WARN] No valid metric columns found for centralized aggregation.")
        return None

    def _to_hashable_scalar(val):
        if isinstance(val, dict):
            for k in ["value", "name", "id", "model", "split"]:
                if k in val and not isinstance(val[k], (dict, list, tuple, set)):
                    return val[k]
            try:
                return json.dumps(val, sort_keys=True)
            except Exception:
                return str(val)
        if isinstance(val, (list, tuple, set)):
            try:
                return json.dumps(list(val), sort_keys=True)
            except Exception:
                return str(val)
        return val

    work = df.copy()
    work["config.model"] = work["config.model"].apply(_to_hashable_scalar)

    split_col = "config.split" if "config.split" in df.columns else None

    if split_col is not None:
        work[split_col] = work[split_col].apply(_to_hashable_scalar)
        per_split = (
            work.groupby(["config.model", split_col], dropna=False)[valid_metrics]
            .mean()
            .reset_index()
        )
    else:
        print("[WARN] 'config.split' not found: using runs directly (no split-level pre-aggregation).")
        per_split = work[["config.model"] + valid_metrics].copy()

    grouped = (
        per_split.groupby(["config.model"])[valid_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    grouped.columns = [
        "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    return grouped

def compute_centralized_task_architecture_tables(df: pd.DataFrame):
    """
    Centralized case across all tasks:
    - average metrics per (task, architecture, split)
    - then compute mean/std across splits per (task, architecture)
    - return a pivot table: rows=task, cols=architecture, cell='RMSE mean ± std | NASA mean ± std'
    """
    required_cols = [
        "config.task",
        "config.model",
        "summary.global/test_rmse",
        "summary.global/test_nasa_score",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing required columns for centralized task table: {missing}")
        return None, None

    def _to_hashable_scalar(val):
        if isinstance(val, dict):
            for k in ["value", "name", "id", "model", "split", "task"]:
                if k in val and not isinstance(val[k], (dict, list, tuple, set)):
                    return val[k]
            try:
                return json.dumps(val, sort_keys=True)
            except Exception:
                return str(val)
        if isinstance(val, (list, tuple, set)):
            try:
                return json.dumps(list(val), sort_keys=True)
            except Exception:
                return str(val)
        return val

    work = df.copy()
    work["task"] = work["config.task"].apply(_to_hashable_scalar)
    work["model_raw"] = work["config.model"].apply(_to_hashable_scalar)
    work["architecture"] = work["model_raw"].map(MODEL_MAP).fillna(work["model_raw"])
    work["summary.global/test_rmse"] = pd.to_numeric(work["summary.global/test_rmse"], errors="coerce")
    work["summary.global/test_nasa_score"] = pd.to_numeric(work["summary.global/test_nasa_score"], errors="coerce")

    split_col = "config.split" if "config.split" in work.columns else None
    if split_col is not None:
        work[split_col] = work[split_col].apply(_to_hashable_scalar)
        per_split = (
            work.groupby(["task", "architecture", split_col], dropna=False)[
                ["summary.global/test_rmse", "summary.global/test_nasa_score"]
            ]
            .mean()
            .reset_index()
        )
    else:
        print("[WARN] 'config.split' not found: using runs directly (no split-level pre-aggregation).")
        per_split = work[["task", "architecture", "summary.global/test_rmse", "summary.global/test_nasa_score"]].copy()

    agg = (
        per_split.groupby(["task", "architecture"])[["summary.global/test_rmse", "summary.global/test_nasa_score"]]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    task_order = sorted(agg["task"].dropna().astype(str).unique().tolist())
    table = pd.DataFrame(index=task_order, columns=ARCHITECTURE_ORDER, dtype=object)

    for _, row in agg.iterrows():
        task = str(row["task"])
        arch = row["architecture"]
        if arch not in table.columns:
            table[arch] = "-"
        rmse_text = _fmt_mean_std(
            row.get("summary.global/test_rmse_mean"),
            row.get("summary.global/test_rmse_std"),
        )
        nasa_text = _fmt_mean_std(
            row.get("summary.global/test_nasa_score_mean"),
            row.get("summary.global/test_nasa_score_std"),
        )
        table.loc[task, arch] = f"{rmse_text} | {nasa_text}"

    table = table.fillna("-")
    return table, agg

def _fmt_mean_std(mean_val, std_val, decimals=4):
    if pd.isna(mean_val):
        return "-"
    if pd.isna(std_val):
        return f"{mean_val:.{decimals}f} ± n/a"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"

def _to_hashable_scalar(val):
    if isinstance(val, dict):
        for k in ["value", "name", "id", "model", "split", "task", "algorithm"]:
            if k in val and not isinstance(val[k], (dict, list, tuple, set)):
                return val[k]
        try:
            return json.dumps(val, sort_keys=True)
        except Exception:
            return str(val)
    if isinstance(val, (list, tuple, set)):
        try:
            return json.dumps(list(val), sort_keys=True)
        except Exception:
            return str(val)
    return val

def compute_task_pivot_tables(df: pd.DataFrame):
    required_cols = [
        "config.algorithm",
        "config.model",
        "config.local_epochs",
        "summary.global/test_rmse",
        "summary.global/test_nasa_score",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing required columns for task pivot tables: {missing}")
        return {}, None

    work = df.copy()
    work["algorithm"] = work["config.algorithm"].map(ALGORITHM_MAP)
    work["architecture"] = work["config.model"].map(MODEL_MAP)
    work["local_epochs"] = pd.to_numeric(work["config.local_epochs"], errors="coerce")

    work["summary.global/test_rmse"] = pd.to_numeric(work["summary.global/test_rmse"], errors="coerce")
    work["summary.global/test_nasa_score"] = pd.to_numeric(work["summary.global/test_nasa_score"], errors="coerce")

    work = work.dropna(subset=["algorithm", "architecture", "local_epochs"])

    split_col = "config.split" if "config.split" in work.columns else None
    if split_col is not None:
        per_split = (
            work.groupby(["algorithm", "architecture", "local_epochs", split_col], dropna=False)[
                ["summary.global/test_rmse", "summary.global/test_nasa_score"]
            ]
            .mean()
            .reset_index()
        )
    else:
        print("[WARN] 'config.split' not found: using runs directly (no split-level pre-aggregation).")
        per_split = work[["algorithm", "architecture", "local_epochs", "summary.global/test_rmse", "summary.global/test_nasa_score"]].copy()

    agg = (
        per_split.groupby(["algorithm", "architecture", "local_epochs"])[
            ["summary.global/test_rmse", "summary.global/test_nasa_score"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    local_epoch_values = sorted(set(agg["local_epochs"].dropna().astype(int).tolist()))
    tables = {}

    for le in local_epoch_values:
        sub = agg[agg["local_epochs"] == le].copy()

        table = pd.DataFrame(index=ARCHITECTURE_ORDER, columns=ALGORITHM_ORDER, dtype=object)

        for _, row in sub.iterrows():
            alg = row["algorithm"]
            arch = row["architecture"]
            if alg not in ALGORITHM_ORDER or arch not in ARCHITECTURE_ORDER:
                continue

            rmse_text = _fmt_mean_std(
                row.get("summary.global/test_rmse_mean"),
                row.get("summary.global/test_rmse_std"),
            )
            nasa_text = _fmt_mean_std(
                row.get("summary.global/test_nasa_score_mean"),
                row.get("summary.global/test_nasa_score_std"),
            )
            table.loc[arch, alg] = f"{rmse_text} | {nasa_text}"

        table = table.fillna("-")
        tables[int(le)] = table

    return tables, agg

def get_metrics_at_steps(run, steps, metric_keys):
    """
    Returns dict: {step: {metric: value, ...}, ...} for exact global-round matches.
    Priority for round key in history row:
      1) round
      2) global_round
      3) global/round
      4) _step (fallback only)
    metric_keys are history metric names (WITHOUT 'summary.').
    """
    steps = set(int(s) for s in steps)
    keys = ["round", "_step"] + metric_keys
    out = {}

    for row in run.scan_history(keys=keys):
        s = None
        for key in ["round", "global_round", "global/round", "_step"]:
            val = row.get(key)
            if val is None or pd.isna(val):
                continue
            try:
                s = int(val)
                break
            except Exception:
                continue

        if s in steps:
            out[s] = {k: row.get(k) for k in metric_keys}

    return out

def compute_stepwise_stats(runs_list, target_steps, metric_keys):
    """
    Builds:
      - df_steps: one row per run x step
      - stats_by_step: aggregated count/mean/std by step
    """
    records = []

    for r in runs_list:
        vals = get_metrics_at_steps(r, target_steps, metric_keys)

        for step in target_steps:
            rec = {
                "run_id": r.id,
                "run_name": r.name,
                "step": step,
            }
            for m in metric_keys:
                rec[m] = vals.get(step, {}).get(m, None)
            records.append(rec)

    df_steps = pd.DataFrame(records)

    # convert metrics to numeric safely
    for m in metric_keys:
        df_steps[m] = pd.to_numeric(df_steps[m], errors="coerce")

    stats_by_step = (
        df_steps.groupby("step")[metric_keys]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    return df_steps, stats_by_step

def _extract_config_id_series(df: pd.DataFrame) -> pd.Series:
    """
    Attempts to extract a numeric config_id from the DataFrame using multiple strategies:
        1) Look for common flat column names like 'config.config_id', 'config_id', etc.
        2) If not found, look for nested dicts in columns like 'config.args' and try to extract config_id from there.
        3) If still not found, return a Series of NA.
    """

    flat_candidates = [
        "config.config_id",
        "config.config-id",
        "config.args.config_id",
        "config.args.config-id",
        "config.arg.config_id",
        "config.arg.config-id",
        "config_id",
        "config-id",
    ]

    for col in flat_candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("Int64")

    nested_candidates = ["config.args", "config.arg", "config"]
    for col in nested_candidates:
        if col in df.columns:
            def _get_id(x):
                if isinstance(x, dict):
                    for k in ["config_id", "config-id"]:
                        if k in x:
                            return x[k]

                    for subk in ["args", "arg"]:
                        sub = x.get(subk)
                        if isinstance(sub, dict):
                            for k in ["config_id", "config-id"]:
                                if k in sub:
                                    return sub[k]
                return pd.NA

            return pd.to_numeric(df[col].apply(_get_id), errors="coerce").astype("Int64")

    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

def backfill_configs_from_json(
    df: pd.DataFrame,
    json_path: str,
    config_id_base: str = "auto",   # "auto", "0", "1"
    target_fields=None,
) -> pd.DataFrame:
    """
    Backfills missing config.* fields in df using a JSON file that maps config.config_id to config values.
        - df: DataFrame with runs data, expected to have 'config.config_id' column (int or str)
        - json_path: path to JSON file containing a list of config dicts (without config.config_id)
        - config_id_base: "0", "1", or "auto" to determine if config.config_id in JSON is 0-based or 1-based
        - target_fields: list of config fields to backfill (e.g. ["model", "algorithm"]). If None, defaults to a common set.
    
    Returns a new DataFrame with missing config fields filled where possible.
    """

    if df is None or df.empty:
        print("[INFO] Empty DataFrame provided. Backfill skipped.")
        return df

    out = df.copy()
    out["config.config_id"] = _extract_config_id_series(out)

    n_with_id = int(out["config.config_id"].notna().sum())
    print(f"[DEBUG] Rows total: {len(out)}")
    print(f"[DEBUG] Rows with detected config_id: {n_with_id}")

    if n_with_id == 0:
        print("[INFO] No config_id detected in any row: backfill skipped.")
        return out

    if target_fields is None:
        target_fields = [
            "task",
            "global_rounds",
            "split",
            "local_epochs",
            "no_clip_rul",
            "model",
            "algorithm",
        ]

    target_cols = [f"config.{f}" for f in target_fields]

    for c in target_cols:
        if c not in out.columns:
            out[c] = pd.NA

    null_like = ["null", "None", "none", "nan", "NaN", ""]
    for c in target_cols:
        out[c] = out[c].replace(null_like, pd.NA)

    need_backfill_mask = out["config.config_id"].notna() & out[target_cols].isna().any(axis=1)
    n_need = int(need_backfill_mask.sum())
    print(f"[DEBUG] Rows needing backfill: {n_need}")

    if n_need == 0:
        print("[INFO] No missing config.* fields to backfill.")
        return out

    path = Path(json_path)
    if not path.exists():
        print(f"[WARN] File JSON not found: {json_path}. Backfill skipped.")
        return out

    with open(path, "r", encoding="utf-8") as f:
        cfg_list = json.load(f)

    if not isinstance(cfg_list, list) or len(cfg_list) == 0:
        print("[WARN] Empty JSON or invalid format (expected a non-empty list). Backfill skipped.")
        return out

    cfg_df_raw = pd.DataFrame(cfg_list)

    json_available_fields = [f for f in target_fields if f in cfg_df_raw.columns]
    if not json_available_fields:
        print("[WARN] None of the target fields are present in the JSON. Backfill skipped.")
        return out

    def build_lookup(base_zero: bool) -> pd.DataFrame:
        cfg_df = cfg_df_raw.copy()

        if base_zero:
            cfg_df.insert(0, "config.config_id", range(len(cfg_df)))
        else:
            cfg_df.insert(0, "config.config_id", range(1, len(cfg_df) + 1))

        keep = ["config.config_id"] + json_available_fields
        cfg_df = cfg_df[keep]
        rename_map = {c: (c if c == "config.config_id" else f"config.{c}") for c in cfg_df.columns}
        cfg_df = cfg_df.rename(columns=rename_map)
        cfg_df["config.config_id"] = pd.to_numeric(cfg_df["config.config_id"], errors="coerce").astype("Int64")

        return cfg_df

    missing_ids = set(out.loc[need_backfill_mask, "config.config_id"].dropna().astype(int).tolist())

    if config_id_base == "0":
        lookup = build_lookup(base_zero=True)
        chosen_base = 0
    elif config_id_base == "1":
        lookup = build_lookup(base_zero=False)
        chosen_base = 1
    else:
        lookup0 = build_lookup(base_zero=True)
        lookup1 = build_lookup(base_zero=False)

        ids0 = set(lookup0["config.config_id"].dropna().astype(int).tolist())
        ids1 = set(lookup1["config.config_id"].dropna().astype(int).tolist())

        hits0 = len(missing_ids & ids0)
        hits1 = len(missing_ids & ids1)

        if hits1 > hits0:
            lookup = lookup1
            chosen_base = 1
        else:
            lookup = lookup0
            chosen_base = 0

        print(f"[INFO] Backfill config: config_id {chosen_base}-based (matches for missing ids: {hits0} with base 0, {hits1} with base 1)")

    merged = out.merge(lookup, on="config.config_id", how="left", suffixes=("", "_json"))

    filled_counter = {}
    for field in json_available_fields:
        col = f"config.{field}"
        col_json = f"{col}_json"

        if col_json not in merged.columns:
            continue

        before_missing = int(merged[col].isna().sum()) if col in merged.columns else len(merged)

        if col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[col_json])
        else:
            merged[col] = merged[col_json]

        after_missing = int(merged[col].isna().sum())
        filled_counter[col] = before_missing - after_missing

    helper_cols = [c for c in merged.columns if c.endswith("_json")]
    if helper_cols:
        merged = merged.drop(columns=helper_cols)

    print("[INFO] Backfill completed. Configs filled from JSON:")
    any_fill = False
    for k, v in filled_counter.items():
        if v > 0:
            print(f"  - {k}: {v}")
            any_fill = True
    if not any_fill:
        print("  (no fills performed: check config_id_base or ID matching)")

    return merged


def main():
    args = build_argparser().parse_args()
    compact_out_df = None

    print("Connecting to W&B")
    api = wandb.Api()

    filters = parse_filters_from_args(args)

    print("Asking for runs with:")
    print(f"  entity/project: {args.entity}/{args.project}")
    print(f"  filters: {filters}")
    print(f"  order: {args.order}")
    print(f"  per_page: {args.per_page}")

    runs = api.runs(
        f"{args.entity}/{args.project}",
        filters=filters if filters else None,
        order=args.order,
        per_page=args.per_page,
    )

    runs_list = list(runs)
    print(f"Runs returned by API: {len(runs_list)}")

    df = aggregate_runs(runs_list)
    print("\nDone! Shape:", df.shape)
    print(df.head())

    if args.missing_configs_json:
        df = backfill_configs_from_json(
            df,
            json_path=args.missing_configs_json,
            config_id_base="0"
        )
        print("\nAfter backfill Shape:", df.shape)
        print(df.head())
    else:
        print("\n[INFO] --missing-configs-json not provided. Backfill skipped.")

    metrics = ["summary.global/test_rmse", "summary.global/test_nasa_score"]
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")


    available_metrics = [m for m in metrics if m in df.columns]
    if available_metrics:
        stats = df[available_metrics].agg(["count", "mean", "std"])
        print("\n=== Final summary stats (mean/std) ===")
        print(stats)
    else:
        print("\n[WARN] Summary metrics not found in df columns.")

    if args.steps and args.algorithm is None and args.model is None and not (args.task is not None and args.local_epochs is not None):
        group_cols = ["config.algorithm"]

        grouped_stats = compute_grouped_summary_stats(
            df=df,
            group_cols=group_cols,
            metric_cols=available_metrics
        )

        if grouped_stats is not None:
            print("\n=== Grouped summary stats by algorithm (count/mean/std, averaged across models) ===")
            print(grouped_stats)

            if args.out:
                grouped_path = (
                    args.out.replace(".csv", "_grouped_by_algorithm.csv")
                    if args.out.endswith(".csv")
                    else args.out + "_grouped_by_algorithm.csv"
                )
                grouped_stats.to_csv(grouped_path, index=False)
                print(f"\nSaved grouped stats CSV to: {grouped_path}")

    elif args.algorithm in {"Centralized_RUL", "Local_RUL"} and args.task is None and not args.steps:
        table, agg_long = compute_centralized_task_architecture_tables(df)

        if table is None:
            print(f"\n[WARN] No task table generated for {args.algorithm}.")
        else:
            print(f"\n=== {args.algorithm} table by task x architecture (split-averaged) ===")
            print("Cell format: RMSE mean ± std | NASA mean ± std")
            print(table)

            if args.out:
                table_path = (
                    args.out.replace(".csv", f"_{args.algorithm}_task_architecture_table.csv")
                    if args.out.endswith(".csv")
                    else args.out + f"_{args.algorithm}_task_architecture_table.csv"
                )
                table.to_csv(table_path, index=True)
                print(f"Saved task table CSV to: {table_path}")

                agg_path = (
                    args.out.replace(".csv", f"_{args.algorithm}_task_architecture_long.csv")
                    if args.out.endswith(".csv")
                    else args.out + f"_{args.algorithm}_task_architecture_long.csv"
                )
                agg_long.to_csv(agg_path, index=False)
                print(f"Saved task long CSV to: {agg_path}")

    elif args.group_by == "architecture" and not args.steps:
        if args.algorithm in {"Centralized_RUL", "Local_RUL"}:
            grouped_title = f"\n=== Grouped summary stats by architecture/model ({args.algorithm}) (count/mean/std) ==="
            grouped_stats = compute_centralized_model_split_stats(
                df=df,
                metric_cols=available_metrics,
            )
        else:
            group_cols = ["config.algorithm", "config.model", "config.local_epochs"]
            grouped_title = "\n=== Grouped summary stats by algorithm + model + local_epochs (count/mean/std) ==="
            grouped_stats = compute_grouped_summary_stats(
                df=df,
                group_cols=group_cols,
                metric_cols=available_metrics
            )

        if grouped_stats is not None:
            print(grouped_title)
            print(grouped_stats)

            if args.out:
                grouped_path = (
                    args.out.replace(".csv", "_grouped_by_architecture.csv")
                    if args.out.endswith(".csv")
                    else args.out + "_grouped_by_architecture.csv"
                )
                grouped_stats.to_csv(grouped_path, index=False)
                print(f"\nSaved grouped stats CSV to: {grouped_path}")

    elif args.group_by == "algorithm_local_epochs" and not args.steps:
        group_cols = ["config.algorithm", "config.local_epochs"]

        grouped_stats = compute_grouped_summary_stats(
            df=df,
            group_cols=group_cols,
            metric_cols=available_metrics
        )

        if grouped_stats is not None:
            print("\n=== Grouped summary stats by algorithm + local_epochs (count/mean/std, averaged across architectures) ===")
            print(grouped_stats)

            if args.out:
                grouped_path = (
                    args.out.replace(".csv", "_grouped_by_algorithm_local_epochs.csv")
                    if args.out.endswith(".csv")
                    else args.out + "_grouped_by_algorithm_local_epochs.csv"
                )
                grouped_stats.to_csv(grouped_path, index=False)
                print(f"\nSaved grouped stats CSV to: {grouped_path}")

    elif args.task is not None and args.algorithm is None and args.model is None and args.local_epochs is None and not args.steps:
        pivot_tables, agg_long = compute_task_pivot_tables(df)

        if not pivot_tables:
            print("\n[WARN] No pivot tables generated for task-only mode.")
        else:
            print("\n=== Task-only tables by local_epochs (rows=architecture, cols=algorithm) ===")
            print("Cell format: RMSE mean ± std | NASA mean ± std")
            for le in sorted(pivot_tables.keys()):
                print(f"\n--- local_epochs = {le} ---")
                print(pivot_tables[le])

            if args.out:
                for le, table in pivot_tables.items():
                    path = (
                        args.out.replace(".csv", f"_local_epochs_{le}_pivot.csv")
                        if args.out.endswith(".csv")
                        else args.out + f"_local_epochs_{le}_pivot.csv"
                    )
                    table.to_csv(path, index=True)
                    print(f"Saved task pivot table CSV to: {path}")

                agg_path = (
                    args.out.replace(".csv", "_task_agg_long.csv")
                    if args.out.endswith(".csv")
                    else args.out + "_task_agg_long.csv"
                )
                if agg_long is not None:
                    agg_long.to_csv(agg_path, index=False)
                    print(f"Saved task long aggregation CSV to: {agg_path}")

    elif args.algorithm is None and args.model is None and args.local_epochs is None:
        group_cols = ["config.algorithm", "config.local_epochs"]

        grouped_stats = compute_grouped_summary_stats(
            df=df,
            group_cols=group_cols,
            metric_cols=available_metrics
        )

        if grouped_stats is not None:
            print("\n=== Grouped summary stats by algorithm + local_epochs (count/mean/std) ===")
            print(grouped_stats)

            if args.out:
                grouped_path = (
                    args.out.replace(".csv", "_grouped_stats.csv")
                    if args.out.endswith(".csv")
                    else args.out + "_grouped_stats.csv"
                )
                grouped_stats.to_csv(grouped_path, index=False)
                print(f"\nSaved grouped stats CSV to: {grouped_path}")

    # Optional: step-wise metrics from history
    if args.steps:
        print(f"\nExtracting history metrics at steps: {args.steps}")

        hist_metrics = ["global/test_nasa_score", "global/test_rmse"]

        df_steps, step_stats = compute_stepwise_stats(
            runs_list=runs_list,
            target_steps=args.steps,
            metric_keys=hist_metrics
        )

        print("\n=== Per-run values at requested steps ===")
        print(df_steps.head())

        print("\n=== Aggregated stats by step (count/mean/std) ===")
        print(step_stats)

        if (
            args.task is not None
            and args.local_epochs is not None
            and args.algorithm is None
            and args.model is None
            and "config.algorithm" in df.columns
        ):
            meta_cols = ["id", "config.algorithm", "config.model"]
            if "config.split" in df.columns:
                meta_cols.append("config.split")

            run_meta = df[meta_cols].drop_duplicates(subset=["id"]).rename(columns={"id": "run_id"})
            df_steps_meta = df_steps.merge(run_meta, on="run_id", how="left")

            df_steps_meta["config.algorithm"] = df_steps_meta["config.algorithm"].apply(_to_hashable_scalar)
            df_steps_meta["config.model"] = df_steps_meta["config.model"].apply(_to_hashable_scalar)
            if "config.split" in df_steps_meta.columns:
                df_steps_meta["config.split"] = df_steps_meta["config.split"].apply(_to_hashable_scalar)
            else:
                df_steps_meta["config.split"] = "all"

            # First average across runs within the same (step, algorithm, architecture, split)
            per_arch_split = (
                df_steps_meta.groupby(["step", "config.algorithm", "config.model", "config.split"])[hist_metrics]
                .mean()
                .reset_index()
            )

            # Then aggregate by (step, algorithm), averaging across architectures and splits
            step_stats_by_algo_avg_arch_split = (
                per_arch_split.groupby(["step", "config.algorithm"])[hist_metrics]
                .agg(["count", "mean", "std"])
                .reset_index()
            )

            flat_cols = []
            for col in step_stats_by_algo_avg_arch_split.columns:
                if isinstance(col, tuple):
                    flat_cols.append("_".join([str(x) for x in col if x]))
                else:
                    flat_cols.append(col)
            step_stats_by_algo_avg_arch_split.columns = flat_cols

            compact_cols = [
                "step",
                "config.algorithm",
                "global/test_rmse_mean",
                "global/test_rmse_std",
                "global/test_nasa_score_mean",
                "global/test_nasa_score_std",
            ]
            compact_out_df = step_stats_by_algo_avg_arch_split[[c for c in compact_cols if c in step_stats_by_algo_avg_arch_split.columns]].copy()

            print("\n=== Aggregated step-wise stats by algorithm (averaged across architecture and split) ===")
            print(step_stats_by_algo_avg_arch_split)

        if (
            args.algorithm is None
            and args.model is None
            and "config.algorithm" in df.columns
            and not (args.task is not None and args.local_epochs is not None)
        ):
            run_meta = df[["id", "config.algorithm"]].drop_duplicates(subset=["id"]).rename(columns={"id": "run_id"})
            df_steps_with_algo = df_steps.merge(run_meta, on="run_id", how="left")
            df_steps_with_algo["config.algorithm"] = df_steps_with_algo["config.algorithm"].apply(_to_hashable_scalar)

            step_stats_by_algo = (
                df_steps_with_algo.groupby(["step", "config.algorithm"])[hist_metrics]
                .agg(["count", "mean", "std"])
                .reset_index()
            )

            flat_cols = []
            for col in step_stats_by_algo.columns:
                if isinstance(col, tuple):
                    flat_cols.append("_".join([str(x) for x in col if x]))
                else:
                    flat_cols.append(col)
            step_stats_by_algo.columns = flat_cols

            print("\n=== Aggregated step-wise stats by algorithm (count/mean/std, averaged across models) ===")
            print(step_stats_by_algo)

        if args.out:
            step_stats_path = args.out.replace(".csv", "_step_stats.csv") if args.out.endswith(".csv") else args.out + "_step_stats.csv"
            step_stats.to_csv(step_stats_path, index=False)
            print(f"\nSaved step-wise stats CSV to: {step_stats_path}")

            if (
                args.task is not None
                and args.local_epochs is not None
                and args.algorithm is None
                and args.model is None
                and "config.algorithm" in df.columns
            ):
                step_stats_avg_arch_split_path = (
                    args.out.replace(
                        ".csv",
                        f"_step_stats_by_algorithm_avg_arch_split_task_{args.task}_le_{args.local_epochs}.csv",
                    )
                    if args.out.endswith(".csv")
                    else args.out + f"_step_stats_by_algorithm_avg_arch_split_task_{args.task}_le_{args.local_epochs}.csv"
                )
                step_stats_by_algo_avg_arch_split.to_csv(step_stats_avg_arch_split_path, index=False)
                print(f"\nSaved step-wise stats by algorithm (avg arch+split) CSV to: {step_stats_avg_arch_split_path}")

            if (
                args.algorithm is None
                and args.model is None
                and "config.algorithm" in df.columns
                and not (args.task is not None and args.local_epochs is not None)
            ):
                step_stats_algo_path = (
                    args.out.replace(".csv", f"_step_stats_by_algorithm_task_{args.task}.csv")
                    if args.out.endswith(".csv")
                    else args.out + f"_step_stats_by_algorithm_task_{args.task}.csv"
                )
                step_stats_by_algo.to_csv(step_stats_algo_path, index=False)
                print(f"\nSaved step-wise stats by algorithm CSV to: {step_stats_algo_path}")

    if args.out:
        if compact_out_df is not None:
            compact_out_df.to_csv(args.out, index=False)
            print(f"\nSaved compact CSV (RMSE/NASA only) to: {args.out}")
        else:
            df.to_csv(args.out, index=False)
            print(f"\nSaved CSV to: {args.out}")

# /home/asorrenti/git/RocketFL/system/compute_aggregated_results.py --task A --global_rounds 100 --local_epochs 10 --steps 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 --missing-configs-json /home/asorrenti/git/RocketFL/sweeps/missing_configs.json --out 1
# compute_aggregated_results.py --task E --global_rounds 100 --missing-configs-json /home/asorrenti/git/RocketFL/sweeps/missing_configs.json --out 1
# compute_aggregated_results.py --task B --global_rounds 100 --algorithm FedCross_RUL --missing-configs-json /home/asorrenti/git/RocketFL/sweeps/missing_configs.json
if __name__ == "__main__":
    main()