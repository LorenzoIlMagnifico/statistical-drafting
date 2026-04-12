"""
Training entry point. Run from the project root:

    python run_training.py FDN Premier
    python run_training.py FDN Premier --top3
    python run_training.py FDN Premier --top3 --no-onnx
    python run_training.py FDN Premier --keep-dataset
"""
import argparse
import os
import sys

# Make relative data paths work regardless of where the script is called from.
os.chdir(os.path.join(os.path.dirname(__file__), "notebooks"))

import statisticaldrafting as sd


def main():
    parser = argparse.ArgumentParser(description="Train a DraftNet model.")
    parser.add_argument("set", help="Set abbreviation, e.g. FDN")
    parser.add_argument("mode", help="Draft mode: Premier, Trad, PickTwo, PickTwoTrad")
    parser.add_argument(
        "--top3",
        action="store_true",
        help="Filter to top 3%% of drafters (winrate>=0.66, n_games>=100)",
    )
    parser.add_argument(
        "--keep-dataset",
        action="store_true",
        help="Skip dataset regeneration if a cached .pth already exists",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Skip ONNX export after training",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Use EmbeddingDraftNet instead of the default flat MLP",
    )
    args = parser.parse_args()

    model_type = "embed" if args.embed else "mlp"
    print(f"Training {args.set} {args.mode} | model={model_type} | top3={args.top3} | overwrite={not args.keep_dataset} | onnx={not args.no_onnx}")
    sys.stdout.flush()

    sd.default_training_pipeline(
        set_abbreviation=args.set,
        draft_mode=args.mode,
        overwrite_dataset=not args.keep_dataset,
        top_drafters_only=args.top3,
        export_onnx=not args.no_onnx,
        model_type=model_type,
    )


if __name__ == "__main__":
    main()
