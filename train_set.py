#!/usr/bin/env python3
"""
Train a draft model for a given MTG set.

Usage:
    python train_set.py FDN
    python train_set.py FDN Premier
    python train_set.py TMT Trad --no-onnx

Data files are looked up in data/17-lands-data/ and copied to data/17lands/ if needed.
"""

import argparse
import os
import shutil
import sys

# Force UTF-8 output on Windows to handle emoji in training scripts
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SRC = os.path.join(SCRIPT_DIR, "data", "17-lands-data")
DATA_DEST = os.path.join(SCRIPT_DIR, "data", "17lands")


def ensure_data_file(set_code: str, draft_mode: str) -> bool:
    """Copy the draft CSV from data/17-lands-data/ to data/17lands/ if not already there."""
    filename = f"draft_data_public.{set_code}.{draft_mode}Draft.csv.gz"
    src = os.path.join(DATA_SRC, filename)
    dest = os.path.join(DATA_DEST, filename)

    if os.path.exists(dest):
        print(f"Data file already present: {dest}")
        return True

    if os.path.exists(src):
        os.makedirs(DATA_DEST, exist_ok=True)
        print(f"Copying {src} -> {dest}")
        shutil.copy2(src, dest)
        return True

    print(f"ERROR: Data file not found.\n  Checked: {src}\n  Checked: {dest}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Train an MTG draft model for a set.")
    parser.add_argument("set_code", help="Set abbreviation (e.g. FDN, TMT)")
    parser.add_argument(
        "draft_mode",
        nargs="?",
        default="Premier",
        help="Draft mode: Premier, Trad, PickTwo, PickTwoTrad (default: Premier)",
    )
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument(
        "--keep-dataset",
        action="store_true",
        help="Skip dataset creation if one already exists for this set/mode",
    )
    args = parser.parse_args()

    set_code = args.set_code.upper()
    draft_mode = args.draft_mode

    print(f"=== Training {set_code} {draft_mode} Draft ===\n")

    if not ensure_data_file(set_code, draft_mode):
        sys.exit(1)

    # default_training_pipeline derives the notebooks/ path from its cwd:
    #   notebooks_dir = os.path.dirname(os.getcwd()) + "/notebooks"
    # so we must run it with cwd = model_refresh/.
    model_refresh_dir = os.path.join(SCRIPT_DIR, "model_refresh")
    os.chdir(model_refresh_dir)

    sys.path.insert(0, SCRIPT_DIR)
    import statisticaldrafting as sd

    training_info = sd.default_training_pipeline(
        set_abbreviation=set_code,
        draft_mode=draft_mode,
        overwrite_dataset=not args.keep_dataset,
        export_onnx=not args.no_onnx,
    )

    print("\n=== Training complete ===")
    print(f"  Experiment : {training_info['experiment_name']}")
    print(f"  Date       : {training_info['training_date']}")
    print(f"  Accuracy   : {training_info['validation_accuracy']:.2f}%")
    print(f"  Train picks: {training_info['training_picks']:,}")
    print(f"  Best epoch : {training_info['num_epochs']}")


if __name__ == "__main__":
    main()
