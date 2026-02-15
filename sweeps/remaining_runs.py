import wandb
import yaml
import json
import itertools
from pathlib import Path
from tqdm import tqdm

############################################
# USER SETTINGS
############################################
ENTITY = "mpennisi"
PROJECT = "RocketFL-system"
OLD_SWEEP_ID = "fd0ybdzq"
ORIGINAL_SWEEP_YAML = "sweeps/methods_grid_1.yaml"

OUTPUT_CONFIGS = "missing_configs.json"
OUTPUT_SWEEP = "new_sweep.yaml"
############################################


def normalize_config(cfg, keys):
    """
    Keep only sweep parameter keys and normalize ordering
    """
    filtered = {k: cfg[k] for k in keys if k in cfg}
    return tuple(sorted(filtered.items()))


def main():

    print("Connecting to W&B...")
    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{OLD_SWEEP_ID}")

    print("Loading original sweep YAML...")
    with open(ORIGINAL_SWEEP_YAML) as f:
        sweep_cfg = yaml.safe_load(f)

    parameters = sweep_cfg["parameters"]
    param_keys = list(parameters.keys())

    ############################################
    # Step 1: Get finished runs
    ############################################
    print("Fetching finished runs...")
    finished_configs = []

    for run in tqdm(sweep.runs, desc="Processing runs"):
        if run.state == "finished":
            finished_configs.append(
                normalize_config(run.config, param_keys)
            )

    finished_set = set(finished_configs)
    print(f"Finished runs found: {len(finished_set)}")

    ############################################
    # Step 2: Rebuild full grid
    ############################################
    print("Reconstructing full grid...")

    value_lists = []
    for key in param_keys:
        param_info = parameters[key]
        if "values" not in param_info:
            raise ValueError(
                f"Parameter '{key}' does not define 'values'. "
                "This script assumes a grid sweep."
            )
        value_lists.append(param_info["values"])

    full_grid = [
        dict(zip(param_keys, combo))
        for combo in itertools.product(*value_lists)
    ]

    print(f"Total grid size: {len(full_grid)}")

    ############################################
    # Step 3: Find missing configs
    ############################################
    print("Computing missing configurations...")

    missing = []
    for cfg in full_grid:
        norm = tuple(sorted(cfg.items()))
        if norm not in finished_set:
            missing.append(cfg)

    print(f"Missing configs: {len(missing)}")

    ############################################
    # Step 4: Save missing configs
    ############################################
    print(f"Saving missing configs to {OUTPUT_CONFIGS}...")
    with open(OUTPUT_CONFIGS, "w") as f:
        json.dump(missing, f, indent=2)

    ############################################
    # Step 5: Create new sweep YAML
    ############################################
    print(f"Creating new sweep file {OUTPUT_SWEEP}...")

    new_sweep = {
        "method": "grid",
        "metric": sweep_cfg.get("metric", {}),
        "parameters": {
            "config_id": {
                "values": list(range(len(missing)))
            }
        }
    }

    with open(OUTPUT_SWEEP, "w") as f:
        yaml.dump(new_sweep, f, sort_keys=False)

    print("Done.")
    print(f"Next steps:")
    print(f"1. Modify your training script to load configs from {OUTPUT_CONFIGS}")
    print(f"2. Launch sweep: wandb sweep {OUTPUT_SWEEP}")


if __name__ == "__main__":
    main()