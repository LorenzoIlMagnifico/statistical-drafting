#!/usr/bin/env python3
"""
Train a LightGBM gradient-boosted tree model for draft pick prediction.

Loads the same preprocessed PickDataset used by the NN, trains a multiclass
LightGBM model, and evaluates with the same accuracy metric.

Usage:
    python train_gbt.py FDN
    python train_gbt.py FDN Premier
    python train_gbt.py FDN --no-train  # evaluate only (requires saved model)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def dataset_to_arrays(dataset):
    """
    Convert a PickDataset into flat numpy feature/label arrays for LightGBM.

    Features: pool + missing + position  (matches DraftNet input, pack excluded)
    Label:    argmax of pick_vector (card index)
    Pack:     returned separately for inference-time masking
    """
    n = len(dataset)
    pool_dim = dataset.pools.shape[1]
    miss_dim = dataset.missing.shape[1]
    pos_dim = dataset.positions.shape[1]

    X = np.concatenate([
        dataset.pools.astype(np.float32),
        dataset.missing.astype(np.float32),
        dataset.positions.astype(np.float32),
    ], axis=1)

    y = np.argmax(dataset.pick_vectors, axis=1).astype(np.int32)
    packs = dataset.packs.astype(bool)

    return X, y, packs


def evaluate_accuracy(model, X, y, packs):
    """Accuracy after masking predictions to cards available in pack."""
    raw_scores = model.predict(X)  # shape (N, n_classes)
    # Mask out cards not in the pack.
    raw_scores[~packs] = -np.inf
    predicted = np.argmax(raw_scores, axis=1)
    acc = np.mean(predicted == y) * 100
    print(f"Validation set pick accuracy = {round(acc, 2)}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_code", help="Set abbreviation (e.g. FDN)")
    parser.add_argument("draft_mode", nargs="?", default="Premier")
    parser.add_argument("--no-train", action="store_true", help="Skip training, load saved model")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    set_code = args.set_code.upper()
    draft_mode = args.draft_mode

    train_path = os.path.join(SCRIPT_DIR, "data", "training_sets", f"{set_code}_{draft_mode}_train.pth")
    val_path   = os.path.join(SCRIPT_DIR, "data", "training_sets", f"{set_code}_{draft_mode}_val.pth")
    model_path = os.path.join(SCRIPT_DIR, "data", "models", f"{set_code}_{draft_mode}_gbt.txt")

    if not os.path.exists(train_path):
        print(f"ERROR: Training set not found at {train_path}")
        print("Run train_set.py first to create the dataset.")
        sys.exit(1)

    print(f"=== GBT Training: {set_code} {draft_mode} ===\n")
    print("Loading datasets...")
    t0 = time.time()
    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset   = torch.load(val_path,   weights_only=False)

    print(f"  Train: {len(train_dataset):,} picks")
    print(f"  Val:   {len(val_dataset):,} picks")

    X_train, y_train, packs_train = dataset_to_arrays(train_dataset)
    X_val,   y_val,   packs_val   = dataset_to_arrays(val_dataset)
    n_classes = len(train_dataset.cardnames)
    print(f"  Features: {X_train.shape[1]}  |  Classes: {n_classes}")
    print(f"  Loaded in {round(time.time() - t0, 1)}s\n")

    if args.no_train:
        if not os.path.exists(model_path):
            print(f"ERROR: No saved model found at {model_path}")
            sys.exit(1)
        print(f"Loading saved model from {model_path}")
        model = lgb.Booster(model_file=model_path)
    else:
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": -1,
        }
        print(f"Training LightGBM with params: {params}\n")

        callbacks = [lgb.log_evaluation(period=50)]
        model = lgb.LGBMClassifier(**params)

        t1 = time.time()
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        print(f"\nTraining complete in {round(time.time() - t1)}s")
        model.booster_.save_model(model_path)
        print(f"Model saved to {model_path}")
        model = model.booster_

    print("\nEvaluating...")
    evaluate_accuracy(model, X_val, y_val, packs_val)


if __name__ == "__main__":
    main()
