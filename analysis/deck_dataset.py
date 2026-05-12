"""
deck_dataset.py

Extracts per-draft deck compositions and win counts from 17lands pick data.

Each row in the 17lands CSV is one pick.  The column `pick_maindeck_rate` is
1.0 when that specific card (in that specific draft) ended up in the main deck,
0.0 when it was sideboarded.  Grouping by `draft_id` and summing the maindecked
picks therefore gives the integer maindeck count for every card in that draft.

Public API
----------
load_deck_dataset(csv_path, cards_path, ...)
    → (deck_matrix, wins, card_names)

    deck_matrix : np.ndarray, shape (N_drafts, N_cards), dtype uint8
        Number of copies of each card in the maindeck.
    wins        : np.ndarray, shape (N_drafts,), dtype int32
        event_match_wins for each draft.
    card_names  : list[str]
        Card name for each column of deck_matrix.

save_deck_dataset(path, deck_matrix, wins, card_names)
    Saves the dataset to a .npz file.

load_saved_deck_dataset(path)
    → (deck_matrix, wins, card_names)
    Loads a previously saved dataset from a .npz file.
"""

import gc
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASIC_LANDS = {"Forest", "Island", "Mountain", "Plains", "Swamp"}


def _pool_cols_to_card_names(columns: pd.Index) -> List[str]:
    """Return the card name for every pool_<card> column, sorted."""
    return sorted(
        col[len("pool_"):] for col in columns if col.startswith("pool_")
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def load_deck_dataset(
    csv_path: str,
    cards_path: Optional[str] = None,
    compression: str = "infer",
    min_maindeck_rate: float = 1.0,
    drop_basics: bool = True,
    chunksize: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read 17lands draft data and return one deck vector + win count per draft.

    Parameters
    ----------
    csv_path : str
        Path to a 17lands draft CSV (plain or .gz).
    cards_path : str, optional
        Path to a set card CSV (e.g. data/cards/TMT.csv).  When provided, the
        deck matrix columns are ordered to match that file's card list, and
        cards not in the set are dropped.  When omitted, columns are derived
        from the CSV itself (alphabetical order, no basics).
    compression : str
        Passed to pd.read_csv.  "infer" detects .gz automatically.
    min_maindeck_rate : float
        Picks with pick_maindeck_rate >= this threshold count as maindecked.
        Keep at 1.0 for exact maindeck reconstruction.
    drop_basics : bool
        If True, basic lands are excluded from the deck vectors.
    chunksize : int
        Rows per chunk when reading the CSV.

    Returns
    -------
    deck_matrix : np.ndarray, shape (N_drafts, N_cards), dtype uint8
    wins        : np.ndarray, shape (N_drafts,), dtype int32
    card_names  : list[str], length N_cards
    """
    # ------------------------------------------------------------------ #
    # 1. Determine canonical card name list                               #
    # ------------------------------------------------------------------ #
    if cards_path is not None:
        card_df = pd.read_csv(cards_path)
        canonical_cards = card_df["name"].tolist()
        if drop_basics:
            canonical_cards = [c for c in canonical_cards if c not in BASIC_LANDS]
    else:
        canonical_cards = None  # will be resolved from first chunk

    # ------------------------------------------------------------------ #
    # 2. Stream through CSV, accumulating per-draft counts               #
    # ------------------------------------------------------------------ #
    # draft_id → {"wins": int, "deck": dict{card_name: count}}
    draft_records: dict = {}

    print(f"Reading {csv_path} in chunks of {chunksize:,}...")
    first_chunk = True

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, compression=compression):
        # Resolve card names from first chunk when no cards_path supplied
        if first_chunk and canonical_cards is None:
            pool_cols = [col for col in chunk.columns if col.startswith("pool_")]
            canonical_cards = sorted(
                col[len("pool_"):] for col in pool_cols
                if (not drop_basics or col[len("pool_"):] not in BASIC_LANDS)
            )
            first_chunk = False

        # Keep only rows where pick was maindecked
        maindecked = chunk[chunk["pick_maindeck_rate"] >= min_maindeck_rate]

        for _, row in maindecked.iterrows():
            draft_id = row["draft_id"]
            card = row["pick"]

            if drop_basics and card in BASIC_LANDS:
                continue

            if draft_id not in draft_records:
                draft_records[draft_id] = {
                    "wins": int(row["event_match_wins"]),
                    "deck": {},
                }

            deck = draft_records[draft_id]["deck"]
            deck[card] = deck.get(card, 0) + 1

        del chunk
        gc.collect()

    print(f"Extracted {len(draft_records):,} drafts.")

    # ------------------------------------------------------------------ #
    # 3. Build dense matrix                                               #
    # ------------------------------------------------------------------ #
    card_index = {name: i for i, name in enumerate(canonical_cards)}
    n_drafts = len(draft_records)
    n_cards = len(canonical_cards)

    deck_matrix = np.zeros((n_drafts, n_cards), dtype=np.uint8)
    wins = np.zeros(n_drafts, dtype=np.int32)

    for i, (draft_id, record) in enumerate(draft_records.items()):
        wins[i] = record["wins"]
        for card_name, count in record["deck"].items():
            if card_name in card_index:
                deck_matrix[i, card_index[card_name]] = count

    print(f"Deck matrix shape: {deck_matrix.shape}")
    print(f"Wins distribution: {dict(zip(*np.unique(wins, return_counts=True)))}")

    return deck_matrix, wins, canonical_cards


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_deck_dataset(
    path: str,
    deck_matrix: np.ndarray,
    wins: np.ndarray,
    card_names: List[str],
) -> None:
    """Save deck_matrix, wins, and card_names to a compressed .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        deck_matrix=deck_matrix,
        wins=wins,
        card_names=np.array(card_names),
    )
    saved_path = path if path.endswith(".npz") else path + ".npz"
    print(f"Saved deck dataset to {saved_path}  ({deck_matrix.shape[0]:,} drafts, {deck_matrix.shape[1]} cards)")


def load_saved_deck_dataset(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a deck dataset previously saved with save_deck_dataset."""
    data = np.load(path, allow_pickle=False)
    deck_matrix = data["deck_matrix"]
    wins = data["wins"]
    card_names = data["card_names"].tolist()
    print(f"Loaded deck dataset from {path}  ({deck_matrix.shape[0]:,} drafts, {deck_matrix.shape[1]} cards)")
    return deck_matrix, wins, card_names


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    csv = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/17-lands-data/draft_data_public.TMT.PremierDraft.csv"
    )
    cards = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "../data/cards/TMT.csv"
    )

    if not os.path.exists(csv):
        print(f"CSV not found: {csv}")
        sys.exit(1)

    deck_matrix, wins, card_names = load_deck_dataset(
        csv_path=csv,
        cards_path=cards if os.path.exists(cards) else None,
    )

    print(f"\nSample deck (draft 0):")
    deck = deck_matrix[0]
    for name, count in zip(card_names, deck):
        if count > 0:
            print(f"  {count}x {name}")
    print(f"Wins: {wins[0]}")

    out = sys.argv[3] if len(sys.argv) > 3 else "deck_dataset_out"
    save_deck_dataset(out, deck_matrix, wins, card_names)

    # Round-trip check
    dm2, w2, cn2 = load_saved_deck_dataset(out + ".npz")
    assert np.array_equal(deck_matrix, dm2) and np.array_equal(wins, w2) and card_names == cn2
    print("Round-trip check passed.")
