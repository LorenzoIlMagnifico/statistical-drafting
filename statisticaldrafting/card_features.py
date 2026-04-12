"""
Load per-card static and aggregate features from 17lands set JSON files.

Feature vector layout (22 dimensions):
  [0]      cmc / 10                          (1)
  [1-5]    colors: W U B R G                 (5, multi-hot)
  [6-12]   types: Creature Instant Sorcery   (7, multi-hot)
           Enchantment Artifact Planeswalker Land
  [13-16]  rarity: common uncommon rare mythic  (4, one-hot)
  [17]     gihwr / 100  (game in-hand win rate, All Decks)
  [18]     ohwr  / 100  (opening hand win rate, All Decks)
  [19]     alsa  / 15   (avg last seen at)
  [20]     ata   / 15   (avg taken at)
  [21]     iwd   / 10   (improvement when drawn, can be negative)

Total: 22 features.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import torch

CARD_FEATURE_DIM = 22

_COLORS   = ["W", "U", "B", "R", "G"]
_TYPES    = ["Creature", "Instant", "Sorcery", "Enchantment",
             "Artifact", "Planeswalker", "Land"]
_RARITIES = ["common", "uncommon", "rare", "mythic"]

_MODE_MAP = {
    "Premier":      "PremierDraft",
    "Trad":         "TradDraft",
    "PickTwo":      "PickTwoDraft",
    "PickTwoTrad":  "PickTwoTradDraft",
}


def _find_data_file(set_abbrev: str, draft_mode: str, data_dir: str) -> Optional[str]:
    mode_str = _MODE_MAP.get(draft_mode, draft_mode)
    path = os.path.join(data_dir, f"{set_abbrev}_{mode_str}_Data.json")
    return path if os.path.exists(path) else None


def _card_feature_vector(card: dict) -> list:
    feat = []

    # cmc (capped at 1.0 for very high CMC cards)
    feat.append(min(card.get("cmc", 0) / 10.0, 1.0))

    # colors — multi-hot over WUBRG
    colors = set(card.get("colors", []))
    for c in _COLORS:
        feat.append(1.0 if c in colors else 0.0)

    # types — multi-hot
    types = set(card.get("types", []))
    for t in _TYPES:
        feat.append(1.0 if t in types else 0.0)

    # rarity — one-hot
    rarity = card.get("rarity", "").lower()
    for r in _RARITIES:
        feat.append(1.0 if rarity == r else 0.0)

    # 17lands aggregate stats (All Decks)
    all_decks = card.get("deck_colors", {}).get("All Decks", {})
    feat.append(all_decks.get("gihwr", 50.0) / 100.0)
    feat.append(all_decks.get("ohwr",  50.0) / 100.0)
    feat.append(all_decks.get("alsa",   7.5) / 15.0)
    feat.append(all_decks.get("ata",    7.5) / 15.0)
    feat.append(all_decks.get("iwd",    0.0) / 10.0)

    assert len(feat) == CARD_FEATURE_DIM, f"Expected {CARD_FEATURE_DIM} features, got {len(feat)}"
    return feat


def load_card_features(
    set_abbrev: str,
    draft_mode: str,
    cardnames: List[str],
    data_dir: str = "../data/set-ratings",
) -> Optional[torch.Tensor]:
    """
    Load per-card feature vectors for the given set and mode.

    Returns a (N, CARD_FEATURE_DIM) float tensor if the JSON is found,
    ordered to match `cardnames`.  Returns None if no data file exists
    (model falls back to pure learned embeddings).
    """
    path = _find_data_file(set_abbrev, draft_mode, data_dir)
    if path is None:
        print(f"No card-feature file found for {set_abbrev}/{draft_mode} in {data_dir}. "
              "Using learned embeddings only.")
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    card_ratings: dict = raw.get("card_ratings", {})

    # Build name -> feature dict, skipping non-primary faces.
    name_to_feat: dict[str, list] = {}
    for card in card_ratings.values():
        if not card.get("isprimarycard", 1):
            continue
        name_to_feat[card["name"]] = _card_feature_vector(card)

    missing_names = [n for n in cardnames if n not in name_to_feat]
    if missing_names:
        print(f"Warning: {len(missing_names)} cards not found in feature data "
              f"(using zeros): {missing_names[:5]}")

    rows = [
        name_to_feat[name] if name in name_to_feat else [0.0] * CARD_FEATURE_DIM
        for name in cardnames
    ]
    return torch.tensor(rows, dtype=torch.float32)
