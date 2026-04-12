#!/usr/bin/env python3
"""
Enrich a PickDataset with card-level and archetype-level features from
the 17lands set ratings JSON (data/set-ratings/<SET>_<MODE>Draft_Data.json).

Per-card features added (7 values per card, shape N_cards x 7):
    gihwr       -- Games In Hand Win Rate (0–100), overall across all decks
    cmc         -- Converted mana cost
    is_W        -- 1 if card has white in its color identity
    is_U        -- blue
    is_B        -- black
    is_R        -- red
    is_G        -- green

Archetype win rates added (10 values, same for every example):
    WU, WB, WR, WG, UB, UR, UG, BR, BG, RG -- two-color pair win rates

The enriched dataset is saved alongside the original:
    data/training_sets/<SET>_<MODE>_train_enriched.pth
    data/training_sets/<SET>_<MODE>_val_enriched.pth

Usage:
    python enrich_dataset.py FDN
    python enrich_dataset.py FDN Premier
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from statisticaldrafting.trainingset import EnhancedPickDataset

ARCHETYPE_PAIRS = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
COLORS = ["W", "U", "B", "R", "G"]
CARD_FEATURE_NAMES = ["gihwr", "cmc", "is_W", "is_U", "is_B", "is_R", "is_G"]


def load_ratings(set_code: str, draft_mode: str) -> dict:
    path = os.path.join(SCRIPT_DIR, "data", "set-ratings", f"{set_code}_{draft_mode}Draft_Data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ratings file not found: {path}\n"
            f"Expected: data/set-ratings/{set_code}_{draft_mode}Draft_Data.json"
        )
    with open(path) as f:
        return json.load(f)


def build_card_features(cardnames: list, ratings: dict) -> np.ndarray:
    """
    Build (N_cards, 7) float32 array ordered by cardnames.
    Cards not found in the ratings JSON get zeros.
    """
    # Build name -> card data lookup.
    name_to_data = {v["name"]: v for v in ratings["card_ratings"].values()}

    n = len(cardnames)
    features = np.zeros((n, len(CARD_FEATURE_NAMES)), dtype=np.float32)

    missing_cards = []
    for i, name in enumerate(cardnames):
        card = name_to_data.get(name)
        if card is None:
            missing_cards.append(name)
            continue

        gihwr = card.get("deck_colors", {}).get("All Decks", {}).get("gihwr", 0.0)
        cmc = float(card.get("cmc", 0))
        colors = set(card.get("colors", []))

        features[i, 0] = gihwr
        features[i, 1] = cmc
        features[i, 2] = float("W" in colors)
        features[i, 3] = float("U" in colors)
        features[i, 4] = float("B" in colors)
        features[i, 5] = float("R" in colors)
        features[i, 6] = float("G" in colors)

    if missing_cards:
        print(f"  Warning: {len(missing_cards)} cards not found in ratings JSON (features set to 0):")
        for name in missing_cards[:10]:
            print(f"    - {name}")
        if len(missing_cards) > 10:
            print(f"    ... and {len(missing_cards) - 10} more")

    return features


def build_archetype_winrates(ratings: dict) -> np.ndarray:
    """
    Build (10,) float32 array of two-color pair win rates in ARCHETYPE_PAIRS order.
    Tries both orderings of each pair (e.g. WG and GW) since the JSON is inconsistent.
    """
    color_ratings = ratings.get("color_ratings", {})
    wr = np.zeros(len(ARCHETYPE_PAIRS), dtype=np.float32)
    for i, pair in enumerate(ARCHETYPE_PAIRS):
        value = color_ratings.get(pair) or color_ratings.get(pair[::-1])
        if value is None:
            print(f"  Warning: no win rate found for archetype {pair}")
        wr[i] = float(value or 0.0)
    return wr


def enrich(base_dataset, ratings: dict) -> "EnhancedPickDataset":
    card_features = build_card_features(base_dataset.cardnames, ratings)
    archetype_wr = build_archetype_winrates(ratings)

    print(f"  Card features shape : {card_features.shape}  (N_cards x {len(CARD_FEATURE_NAMES)})")
    print(f"  Archetype win rates : {dict(zip(ARCHETYPE_PAIRS, archetype_wr.round(2).tolist()))}")

    return EnhancedPickDataset(base_dataset, card_features, archetype_wr)


def main():
    parser = argparse.ArgumentParser(description="Enrich a PickDataset with card and archetype features.")
    parser.add_argument("set_code", help="Set abbreviation (e.g. FDN)")
    parser.add_argument("draft_mode", nargs="?", default="Premier")
    args = parser.parse_args()

    set_code = args.set_code.upper()
    draft_mode = args.draft_mode

    training_sets_dir = os.path.join(SCRIPT_DIR, "data", "training_sets")
    train_path = os.path.join(training_sets_dir, f"{set_code}_{draft_mode}_train.pth")
    val_path   = os.path.join(training_sets_dir, f"{set_code}_{draft_mode}_val.pth")
    out_train  = os.path.join(training_sets_dir, f"{set_code}_{draft_mode}_train_enriched.pth")
    out_val    = os.path.join(training_sets_dir, f"{set_code}_{draft_mode}_val_enriched.pth")

    for p in (train_path, val_path):
        if not os.path.exists(p):
            print(f"ERROR: Dataset not found: {p}")
            print("Run train_set.py first to create the base dataset.")
            sys.exit(1)

    print(f"=== Enriching {set_code} {draft_mode} dataset ===\n")

    print("Loading ratings JSON...")
    ratings = load_ratings(set_code, draft_mode)
    meta = ratings.get("meta", {})
    print(f"  Collection date : {meta.get('collection_date', 'unknown')}")
    print(f"  Date range      : {meta.get('start_date')} to {meta.get('end_date')}")
    print(f"  Cards in JSON   : {len(ratings['card_ratings'])}")
    print()

    print("Loading train dataset...")
    train_dataset = torch.load(train_path, weights_only=False)
    print(f"  {len(train_dataset):,} examples, {len(train_dataset.cardnames)} cards")
    train_enriched = enrich(train_dataset, ratings)
    torch.save(train_enriched, out_train)
    print(f"  Saved -> {out_train}\n")

    print("Loading val dataset...")
    val_dataset = torch.load(val_path, weights_only=False)
    print(f"  {len(val_dataset):,} examples")
    val_enriched = enrich(val_dataset, ratings)
    torch.save(val_enriched, out_val)
    print(f"  Saved -> {out_val}\n")

    print("Done.")
    print(f"\nFeature layout per example (in addition to existing pool/pack/position/missing):")
    print(f"  card_features  : ({len(train_dataset.cardnames)}, {len(CARD_FEATURE_NAMES)})  -- {CARD_FEATURE_NAMES}")
    print(f"  archetype_wr   : (10,)  -- {ARCHETYPE_PAIRS}")


if __name__ == "__main__":
    main()
