"""Print a table of rho_per_example and p0_empirical from rho_sweep results.

Usage:
  uv run python -m src.scripts.adhoc.table_rho_p0
  uv run python -m src.scripts.adhoc.table_rho_p0 --root results/rho_sweep
  uv run python -m src.scripts.adhoc.table_rho_p0 --metric rho   # rho only
  uv run python -m src.scripts.adhoc.table_rho_p0 --csv          # machine-readable
  uv run python -m src.scripts.adhoc.table_rho_p0 --update-csv   # sync results into rho_sweep_rho.csv / rho_sweep_baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASK_ORDER = [
    "sst2", "sst5", "rte", "boolq", "mnli", "cb", "wsc", "wic", "copa", "trec",
    "squad", "drop", "record", "gsm8k", "math500", "countdown",
]


def load_results(root: Path) -> dict[str, dict[str, dict]]:
    """Return {model_name: {task_name: result_dict}}."""
    data: dict[str, dict[str, dict]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        data[model] = {}
        for json_file in model_dir.glob("*.json"):
            task = json_file.stem
            try:
                with open(json_file) as f:
                    data[model][task] = json.load(f)
            except Exception:
                pass
    return data


def fmt(val, width=6) -> str:
    if val is None:
        return " " * width
    if isinstance(val, float) and (val != val):  # nan
        return f"{'nan':>{width}}"
    return f"{val:>{width}.3f}"


def print_table(data: dict, metric: str, csv: bool):
    models = sorted(data.keys())
    tasks  = [t for t in TASK_ORDER if any(t in data[m] for m in models)]
    # add any tasks not in TASK_ORDER
    extra  = sorted({t for m in models for t in data[m]} - set(TASK_ORDER))
    tasks += extra

    if csv:
        header = ["model", "task", "rho_per_example", "p0_empirical"]
        print(",".join(header))
        for model in models:
            for task in tasks:
                r = data[model].get(task)
                if r is None:
                    continue
                rho = r.get("rho_per_example")
                p0  = r.get("p0_empirical")
                print(f"{model},{task},{rho},{p0}")
        return

    col_w   = 7   # value column width
    task_w  = 11  # task name column width
    model_w = max(len(m) for m in models) + 2

    if metric in ("rho", "both"):
        _print_single_table(data, models, tasks, "rho_per_example", "ρ (per-example)", col_w, task_w, model_w)
    if metric in ("p0", "both"):
        if metric == "both":
            print()
        _print_single_table(data, models, tasks, "p0_empirical", "p₀ (base acc)", col_w, task_w, model_w)


def _print_single_table(data, models, tasks, field, title, col_w, task_w, model_w):
    sep = "-" * (task_w + 2 + len(models) * (col_w + 1))
    print(f"\n{title}")
    print(sep)
    header = f"{'task':<{task_w}}  " + "  ".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print(sep)
    for task in tasks:
        row_vals = [data[m].get(task, {}).get(field) for m in models]
        if all(v is None for v in row_vals):
            continue
        row = f"{task:<{task_w}}  " + "  ".join(fmt(v, col_w) for v in row_vals)
        print(row)
    print(sep)


def update_csv(data: dict, csv_rho: Path, csv_p0: Path):
    """Upsert results from `data` into the two wide-format CSV files."""
    for csv_path, field in [(csv_rho, "rho_per_example"), (csv_p0, "p0_empirical")]:
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}  (create it first)")

        # Read existing CSV into {task: {model: value}}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = {row["task"]: dict(row) for row in reader}

        # Determine full model column set: existing columns (minus "task") + any new models
        existing_models = [c for c in columns if c != "task"]
        new_models = [m for m in sorted(data.keys()) if m not in existing_models]
        all_models = existing_models + new_models

        # Ensure all task rows exist
        for task in TASK_ORDER:
            if task not in rows:
                rows[task] = {"task": task, **{m: "" for m in all_models}}

        # Write new values (overwrite existing cell if result now available)
        updated = skipped = 0
        for model, tasks in data.items():
            for task, result in tasks.items():
                if task not in rows:
                    rows[task] = {"task": task, **{m: "" for m in all_models}}
                val = result.get(field)
                if val is None or (isinstance(val, float) and val != val):
                    cell = ""
                else:
                    cell = f"{val:.4f}"
                old = rows[task].get(model, "")
                rows[task][model] = cell
                if cell and cell != old:
                    updated += 1
                else:
                    skipped += 1

        # Write back in task order
        ordered_tasks = [t for t in TASK_ORDER if t in rows]
        ordered_tasks += sorted(t for t in rows if t not in TASK_ORDER)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task"] + all_models, extrasaction="ignore")
            writer.writeheader()
            for task in ordered_tasks:
                writer.writerow({**{m: "" for m in all_models}, **rows[task]})

        print(f"  {csv_path.name}: {updated} cells updated, {skipped} unchanged")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",       default="results/rho_sweep", help="Sweep output directory")
    p.add_argument("--metric",     default="both", choices=["rho", "p0", "both"])
    p.add_argument("--csv",        action="store_true", help="Output CSV instead of pretty table")
    p.add_argument("--update-csv", action="store_true",
                   help="Upsert new results into rho_sweep_rho.csv / rho_sweep_baseline.csv")
    p.add_argument("--csv-dir",    default="results",
                   help="Directory containing the two CSV files (default: results/)")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    data = load_results(root)
    if not data:
        raise SystemExit("No results found.")

    if args.update_csv:
        csv_dir = Path(args.csv_dir)
        print(f"Updating CSVs in {csv_dir}/")
        update_csv(data, csv_dir / "rho_sweep_rho.csv", csv_dir / "rho_sweep_baseline.csv")
    else:
        print_table(data, args.metric, args.csv)


if __name__ == "__main__":
    main()
