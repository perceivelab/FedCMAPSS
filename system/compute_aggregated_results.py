import wandb
import argparse
import pandas as pd

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Aggregate W&B runs for a project, with optional filters on config keys."
    )

    # Output
    parser.add_argument("--out", default=None, help="Optional output CSV path (e.g. runs.csv)")

    # WandB
    parser.add_argument("--entity", default="bomber-team", help="W&B entity/team")
    parser.add_argument("--project", default="Rocket-FL", help="W&B project")

    # Basic run selection
    parser.add_argument("--state", default="finished",
                        help="Filter by run state (finished, running, crashed, etc.)")
    parser.add_argument("--order", default="-created_at",
                        help="Order string for W&B API (default: -created_at)")
    parser.add_argument("--per-page", type=int, default=200,
                        help="Pagination size for W&B API (default: 200)")

    # Experiment filters
    parser.add_argument("--algorithm", type=str,
        choices=["FedAvg_RUL", "SCAFFOLD_RUL", "FedProx_RUL", "FedDyn_RUL", "FedCross_RUL"],
        default="FedAvg_RUL",
        help="Filter by config.algorithm"
    )
    parser.add_argument("--model", type=str,
        choices=["LSTM_RUL", "AFT_RUL", "AttBiGRU_RUL", "RNN_RUL", "Chen_CNN_RUL"],
        default="LSTM_RUL",
        help="Filter by config.model"
    )
    parser.add_argument("--task", type=str, default=None, help="Filter by config.task")
    parser.add_argument("--global_rounds", type=int, default=None, help="Filter by config.global_rounds")
    parser.add_argument("--local_epochs", type=int, choices=[1, 5, 10], default=None, help="Filter by config.local_epochs")

    # Boolean flag
    parser.add_argument("--no_clip_rul", action="store_true",
                        help="If set, filter config.no_clip_rul=True")

    return parser


def parse_filters_from_args(args) -> dict:
    filters = {}

    # run state
    if args.state:
        filters["state"] = args.state

    # config filters (solo se non None)
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
        rows.append({
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "created_at": r.created_at,
            "tags": list(r.tags),
            **{f"config.{k}": v for k, v in dict(r.config).items()},
            **{f"summary.{k}": v for k, v in dict(r.summary).items()},
        })
    return pd.DataFrame(rows)


def main():
    args = build_argparser().parse_args()

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

    df = aggregate_runs(runs)
    print("\nDone! Shape:", df.shape)
    print(df.head())

    metrics = ["summary.global/test_nasa_score", "summary.global/test_rmse"]

    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")


    stats = df[metrics].agg(["mean", "std"])
    print("\n=== Stats (mean and std) ===")
    print(stats)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved CSV to: {args.out}")


if __name__ == "__main__":
    main()