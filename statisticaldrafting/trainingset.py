import gc
import math
import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


def remove_basics(draft_chunk: pd.DataFrame) -> pd.DataFrame:
    # Remove basic lands from raw 17lands dataset
    basic_names = ["Forest", "Island", "Mountain", "Plains", "Swamp"]
    columns_to_drop = ["pack_card_" + b for b in basic_names] + [
        "pool_" + b for b in basic_names
    ]

    # If basics are in set, drop them: 
    if len(set(draft_chunk.columns).intersection(columns_to_drop)) > 0:
        draft_chunk = draft_chunk.drop(columns=columns_to_drop)
        draft_chunk = draft_chunk[~draft_chunk["pick"].isin(basic_names)]

    return draft_chunk


class PickDataset(Dataset):
    def __init__(self, pools, packs, pick_vectors, cardnames, rarities, positions=None, missing=None):
        self.pools = pools
        self.packs = packs
        self.pick_vectors = pick_vectors
        self.cardnames = cardnames
        self.rarities = rarities
        # positions: float32 array of shape (N, 2) with [pack_number/2, pick_number/14]
        # Falls back to zeros for datasets created before this feature was added.
        if positions is not None:
            self.positions = positions
        else:
            self.positions = np.zeros((len(pools), 2), dtype=np.float32)
        # missing: float32 array of shape (N, len(cardnames)) — cards seen in prior packs but not picked.
        # Falls back to zeros if not provided (i.e. no missing-card signal).
        if missing is not None:
            self.missing = missing
        else:
            self.missing = np.zeros_like(pools, dtype=np.float32)

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.pools[index]),
            torch.from_numpy(self.packs[index]),
            torch.from_numpy(self.pick_vectors[index]),
            torch.from_numpy(self.positions[index]),
            torch.from_numpy(self.missing[index]),
        )


class EnhancedPickDataset(Dataset):
    """
    PickDataset augmented with static card features and archetype win rates.

    __getitem__ returns:
        pool, pack, pick_vector, position, missing  -- same as PickDataset
        card_features   -- (N_cards, 7) float32 tensor, same for every example
        archetype_wr    -- (10,) float32 tensor, same for every example
    """

    def __init__(self, base_dataset, card_features: np.ndarray, archetype_wr: np.ndarray):
        self.base = base_dataset
        self.card_features = torch.from_numpy(card_features.astype(np.float32))  # (N_cards, 7)
        self.archetype_wr  = torch.from_numpy(archetype_wr.astype(np.float32))   # (10,)
        self.cardnames = base_dataset.cardnames
        self.rarities  = base_dataset.rarities

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        pool, pack, pick_vec, position, missing = self.base[index]
        return pool, pack, pick_vec, position, missing, self.card_features, self.archetype_wr


def create_card_csv(
    set_abbreviation: str,
    cardnames: List[str],
    data_folder_cards: str = "../data/cards/",
    reprocess: bool = False,
) -> None:
    """
    Creates a csv describing draftable cards, including out-of-set additions.
    """
    # Use existing cardname file if it exists.
    set_card_path = data_folder_cards + set_abbreviation + ".csv"
    if os.path.exists(set_card_path) and reprocess is False:
        print(f"Using existing cardname file, {set_card_path}")
        return

    # This file should be most recent list of cards from 17lands.
    df = pd.read_csv(data_folder_cards + "/cards.csv")

    # Check that set abbreviation is valid.
    if set_abbreviation not in df["expansion"].unique():
        raise Exception(
            f"{set_abbreviation} not found in card list. Consider choosing a new set abbreviation or downloading a new list of cards from https://www.17lands.com/public_datasets"
        )

    # Filter down to cards in set.
    df = df[df["name"].isin(cardnames)]
    df = df[df["is_booster"]]

    # Prioritize cards from the current expansion.
    df["is_target_expansion"] = df["expansion"] == set_abbreviation
    df = df.groupby("name").last()

    # Mark non-expansion cards with "special" rarity
    df.loc[~df["is_target_expansion"], "rarity"] = "special"
    df = df.reset_index()[["name", "rarity", "color_identity"]]

    # Fix color identity
    df["color_identity"] = df["color_identity"].fillna("Colorless")
    df["color_identity"] = df["color_identity"].apply(
        lambda x: "Multicolor" if (len(x) > 1 and x != "Colorless") else x
    )

    # Write csv with cards in set.
    df.to_csv(set_card_path, index=False)
    print(f"Created new cardname file for {set_abbreviation}, {set_card_path}")

def get_min_winrate(n_games: int,
                    p: float = 0.55, 
                    stdev: float = 1.96) -> float:
    """
    Returns minimum winrate that true winrate > p

    For the default stdev=1.96, this is 95% confident
    """
    return p + stdev * math.sqrt(n_games * p * (1 - p)) / n_games

def create_enriched_csv(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    overwrite: bool = False,
    data_folder_17lands: str = "../data/17lands/",
) -> str:
    """
    Creates an enriched version of the 17lands draft CSV with missing-card columns.

    For each pick, adds 'missing_<cardname>' columns: cards that appeared in prior
    packs of the same booster (within this draft) but were not taken by this player.
    These represent cards that other drafters in the pod have picked up.

    Missing resets at the start of each booster pack (packs 0/1/2 are independent
    because they travel in different directions around the table).

    The resulting file is saved as:
        draft_data_public.<SET>.<MODE>Draft.enriched.csv.gz
    and is consumed by create_dataset(use_missing_cards=True).
    """
    input_path = f"{data_folder_17lands}draft_data_public.{set_abbreviation}.{draft_mode}Draft.csv.gz"
    output_path = f"{data_folder_17lands}draft_data_public.{set_abbreviation}.{draft_mode}Draft.enriched.csv.gz"

    if not overwrite and os.path.exists(output_path):
        print(f"Enriched CSV already exists: {output_path}")
        return output_path

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, compression="gzip")
    df = remove_basics(df)

    pack_cols = sorted([col for col in df.columns if col.startswith("pack_card_")])
    pool_cols = sorted([col for col in df.columns if col.startswith("pool_")])
    card_names = [col[len("pack_card_"):] for col in pack_cols]

    print(f"Found {len(card_names)} cards, {len(df):,} rows across {df['draft_id'].nunique():,} drafts.")

    # Sort by draft and pick order so we can iterate sequentially per draft.
    df = df.sort_values(["draft_id", "pack_number", "pick_number"]).reset_index(drop=True)

    # Pre-extract numpy arrays for performance.
    pack_arr = df[pack_cols].values.astype(np.uint8)
    pool_arr = df[pool_cols].values.astype(np.uint8)
    draft_ids = df["draft_id"].values
    pack_numbers = df["pack_number"].values

    # Compute missing vectors row by row.
    print("Computing missing card vectors...")
    t0 = time.time()
    n_cards = len(card_names)
    missing_data = np.zeros((len(df), n_cards), dtype=np.uint8)

    cumulative_seen = np.zeros(n_cards, dtype=np.uint8)
    current_draft = None
    current_pack_number = None

    for i in range(len(df)):
        draft_id = draft_ids[i]
        pack_number = pack_numbers[i]

        # Reset on new draft or new booster pack within the same draft.
        if draft_id != current_draft or pack_number != current_pack_number:
            cumulative_seen[:] = 0
            current_draft = draft_id
            current_pack_number = pack_number

        # Missing = cards seen in prior packs of this booster, not currently in pool.
        missing_data[i] = cumulative_seen & ~pool_arr[i].astype(bool)

        # Accumulate current pack into seen.
        cumulative_seen |= pack_arr[i]

        if i % 100000 == 0 and i > 0:
            print(f"  {i:,} / {len(df):,} rows, t={round(time.time() - t0, 1)}s")

    print(f"Missing vectors computed in {round(time.time() - t0, 1)}s.")

    # Attach missing columns and save.
    missing_cols = [f"missing_{name}" for name in card_names]
    df = pd.concat([df, pd.DataFrame(missing_data, columns=missing_cols)], axis=1)

    print(f"Saving enriched CSV to {output_path}...")
    df.to_csv(output_path, index=False, compression="gzip")
    print(f"Done. {len(df):,} rows saved to {output_path}")

    return output_path


def create_dataset(
    set_abbreviation: str,
    draft_mode: str = "Premier",
    overwrite: bool = False,
    omit_first_days: int = 2,
    train_fraction: float = 0.8,
    use_missing_cards: bool = False,
    top_drafters_only: bool = False,
    data_folder_17lands: str = "../data/17lands/",
    data_folder_training_set: str = "../data/training_sets/",
    data_folder_cards: str = "../data/cards/",
) -> Tuple[str, str]:
    """
    Creates clean training and validation datasets from raw 17lands data.

    Args:
        set_abbreviation (str): Three letter abbreviation of set to create training set of.
        draft_mode (str): Use either "Premier", "Trad", "PickTwo", or "PickTwoTrad" draft data.
        overwrite (bool): If False, won't overwrite an existing dataset for the set and draft mode.
        omit_first_days (int): Omit this many days from the beginning of the dataset.
        use_missing_cards (bool): If True, read missing-card columns from the enriched CSV
            produced by create_enriched_csv(). The enriched file must exist already.
        train_fraction (float): Fraction of dataset to use for training.
        data_folder_17lands (str): Folder where raw 17lands files are stored.
        data_folder_training_set (str): Folder where processed training & validation sets are stored.
        data_folder_cards (str): Folder where card info is stored.
    """
    # Check if training set exists.
    dataset_suffix = "_top3pct" if top_drafters_only else ""
    train_filename = f"{set_abbreviation}_{draft_mode}{dataset_suffix}_train.pth"
    val_filename = f"{set_abbreviation}_{draft_mode}{dataset_suffix}_val.pth"
    train_path = data_folder_training_set + train_filename
    val_path = data_folder_training_set + val_filename
    if overwrite == False and os.path.exists(train_path) and os.path.exists(val_path):
        print("Training and validation sets already exist. Skipping.")
        return train_path, val_path

    # Choose input CSV: enriched (with missing columns) or raw.
    base_csv = f"{data_folder_17lands}draft_data_public.{set_abbreviation}.{draft_mode}Draft.csv.gz"
    enriched_csv = f"{data_folder_17lands}draft_data_public.{set_abbreviation}.{draft_mode}Draft.enriched.csv.gz"
    if use_missing_cards:
        if not os.path.exists(enriched_csv):
            raise FileNotFoundError(
                f"Enriched CSV not found: {enriched_csv}\n"
                f"Run create_enriched_csv('{set_abbreviation}', '{draft_mode}') first."
            )
        csv_path = enriched_csv
        print(f"Using enriched input file {csv_path}")
    else:
        csv_path = base_csv
        if os.path.exists(csv_path):
            print(f"Using input file {csv_path}")
        else:
            print(f"Did not find file {csv_path}")

    # Check if this is a PickTwoDraft mode
    is_picktwo_mode = draft_mode in ["PickTwo", "PickTwoTrad"]
    if is_picktwo_mode:
        print(f"Detected PickTwoDraft mode: {draft_mode}")

    # Initialization on a single chunk.
    for draft_chunk in pd.read_csv(csv_path, chunksize=10000, compression="gzip"):

        # Remove basics.
        draft_chunk = remove_basics(draft_chunk)

        # Get date 1 week after start of draft (assumes drafts sorted by draft time).
        first_date_str = draft_chunk["draft_time"].min()
        first_date_obj = datetime.strptime(first_date_str, "%Y-%m-%d %H:%M:%S")
        min_date_obj = first_date_obj + timedelta(days=omit_first_days)
        min_date_str = min_date_obj.strftime("%Y-%m-%d %H:%M:%S")

        # Get cardnames and ids.
        pack_cols = [col for col in draft_chunk.columns if col.startswith("pack_card")]
        cardnames = [col[10:] for col in sorted(pack_cols)]
        class_to_index = {cls: idx for idx, cls in enumerate(cardnames)}

        print("Completed initialization.")
        
        # Verify that pick_2 column exists for PickTwoDraft modes
        if is_picktwo_mode:
            if "pick_2" not in draft_chunk.columns:
                raise Exception(f"pick_2 column not found in {draft_mode} data. Expected for PickTwoDraft modes.")
            print("Confirmed pick_2 column exists")
        
        break

    # Process full input csv in chunks with immediate train/val split to minimize memory usage.
    # This approach never holds more than one chunk in memory at a time.
    all_data_train = {'pools': [], 'packs': [], 'picks': [], 'positions': [], 'missing': []}
    all_data_val = {'pools': [], 'packs': [], 'picks': [], 'positions': [], 'missing': []}

    chunk_size = 10000  # Reduced from 100000 for CI memory constraints
    t0 = time.time()
    total_examples = 0

    for i, draft_chunk in enumerate(
        pd.read_csv(csv_path, chunksize=chunk_size, compression="gzip")
    ):
        # Remove basics.
        draft_chunk = remove_basics(draft_chunk)

        # # Only keep drafts with maximum # of wins.
        # min_match_wins = 7 if draft_mode == "Premier" else 3
        # draft_chunk = draft_chunk[draft_chunk["event_match_wins"] >= min_match_wins]
        # print(f"Filtering by match wins >= {min_match_wins}")

        # Omit first days.
        draft_chunk = draft_chunk[draft_chunk["draft_time"] >= min_date_str]
        # print("Filtering chunk by date")

        # Require 95% confidence that winrate >= 0.54.
        min_winrate = draft_chunk["user_n_games_bucket"].apply(get_min_winrate, p=0.55, stdev=1.5)
        draft_chunk = draft_chunk[draft_chunk["user_game_win_rate_bucket"] >= min_winrate]
        # print("Filtering chunk by confidence winrate >= 0.55")

        # Top 3% of drafters.
        if top_drafters_only:
            draft_chunk = draft_chunk[draft_chunk["user_game_win_rate_bucket"] >= 0.66]
            draft_chunk = draft_chunk[draft_chunk["user_n_games_bucket"] >= 100]

        # Extract packs.
        pack_chunk = draft_chunk[sorted(pack_cols)].astype(bool)

        # Extract pools.
        pool_cols = [col for col in draft_chunk.columns if col.startswith("pool_")]
        pool_chunk = draft_chunk[sorted(pool_cols)].astype(np.uint8)

        # Extract draft position: pack_number in [0,2], pick_number in [1,14].
        # Normalize to [0, 1] so values are on a similar scale to the pool vector.
        position_chunk = np.stack([
            draft_chunk["pack_number"].values.astype(np.float32) / 2.0,
            draft_chunk["pick_number"].values.astype(np.float32) / 14.0,
        ], axis=1)

        # Extract missing cards (only present in enriched CSV).
        if use_missing_cards:
            missing_cols = sorted([col for col in draft_chunk.columns if col.startswith("missing_")])
            missing_chunk = draft_chunk[missing_cols].values.astype(np.float32)
        else:
            missing_chunk = np.zeros((len(draft_chunk), len(cardnames)), dtype=np.float32)

        # Extract picks.
        if is_picktwo_mode:
            # For PickTwoDraft modes, generate two examples per row: one for pick and one for pick_2
            # We need to duplicate the pack and pool data for each pick
            pick_chunk = np.zeros((len(draft_chunk) * 2, len(cardnames)), dtype=bool)

            # Process pick column
            for j, item in enumerate(draft_chunk["pick"]):
                pick_chunk[j * 2, class_to_index[item]] = True

            # Process pick_2 column
            for j, item in enumerate(draft_chunk["pick_2"]):
                pick_chunk[j * 2 + 1, class_to_index[item]] = True

            # Duplicate pack, pool, position, and missing data for the second example
            pack_chunk = pd.concat([pack_chunk, pack_chunk], ignore_index=True)
            pool_chunk = pd.concat([pool_chunk, pool_chunk], ignore_index=True)
            position_chunk = np.vstack([position_chunk, position_chunk])
            missing_chunk = np.vstack([missing_chunk, missing_chunk])
        else:
            # Standard processing for non-PickTwoDraft modes
            pick_chunk = np.zeros((len(draft_chunk), len(cardnames)), dtype=bool)
            for j, item in enumerate(draft_chunk["pick"]):
                pick_chunk[j, class_to_index[item]] = True

        # Immediately split this chunk into train/val to avoid accumulating all chunks
        if len(pick_chunk) > 1:
            chunk_pools = pool_chunk.values
            chunk_packs = pack_chunk.values

            pools_train, pools_test, packs_train, packs_test, picks_train, picks_test, pos_train, pos_test, miss_train, miss_test = train_test_split(
                chunk_pools, chunk_packs, pick_chunk, position_chunk, missing_chunk, test_size=0.2, random_state=42
            )

            all_data_train['pools'].append(pools_train)
            all_data_train['packs'].append(packs_train)
            all_data_train['picks'].append(picks_train)
            all_data_train['positions'].append(pos_train)
            all_data_train['missing'].append(miss_train)
            all_data_val['pools'].append(pools_test)
            all_data_val['packs'].append(packs_test)
            all_data_val['picks'].append(picks_test)
            all_data_val['positions'].append(pos_test)
            all_data_val['missing'].append(miss_test)

            total_examples += len(pick_chunk)

        # Free chunk memory immediately and force garbage collection
        del draft_chunk, pack_chunk, pool_chunk, pick_chunk, position_chunk, missing_chunk
        gc.collect()

        if i % 10 == 0:
            examples_loaded = total_examples
            print(f"Processed {examples_loaded} picks, t=", round(time.time() - t0, 1), "s")

    print(f"Loaded and split all draft data ({total_examples} total examples).")

    # Make sure we have a card csv.
    create_card_csv(
        set_abbreviation=set_abbreviation, cardnames=cardnames, data_folder_cards=data_folder_cards
    )

    # Get rarities for set.
    rarities = pd.read_csv(data_folder_cards + set_abbreviation + ".csv")["rarity"].tolist() #TODO check if sorted.

    # Final concatenation of train/val splits
    print("Concatenating train/validation splits...")
    pools_train = np.vstack(all_data_train['pools'])
    packs_train = np.vstack(all_data_train['packs'])
    picks_train = np.vstack(all_data_train['picks'])
    positions_train = np.vstack(all_data_train['positions'])
    missing_train = np.vstack(all_data_train['missing'])
    pools_test = np.vstack(all_data_val['pools'])
    packs_test = np.vstack(all_data_val['packs'])
    picks_test = np.vstack(all_data_val['picks'])
    positions_test = np.vstack(all_data_val['positions'])
    missing_test = np.vstack(all_data_val['missing'])

    pick_train_dataset = PickDataset(
        pools_train, packs_train, picks_train, cardnames, rarities, positions_train, missing_train
    )
    pick_val_dataset = PickDataset(
        pools_test, packs_test, picks_test, cardnames, rarities, positions_test, missing_test
    )

    # Free memory from intermediate arrays
    del all_data_train, all_data_val
    del pools_train, packs_train, picks_train, positions_train, missing_train
    del pools_test, packs_test, picks_test, positions_test, missing_test
    gc.collect()

    # Previous train/test split - segmented by time. 
    # tsize = round(len(pools) * train_fraction)
    # pick_train_dataset = PickDataset(
    #     pools[:tsize].values, packs[:tsize].values, picks[:tsize], cardnames, rarities
    # )
    # pick_val_dataset = PickDataset(
    #     pools[tsize:].values, packs[tsize:].values, picks[tsize:], cardnames, rarities
    # )

    # Serialize updated datasets.
    if not os.path.exists(data_folder_training_set):
        os.makedirs(data_folder_training_set)

    # Write datasets.
    torch.save(pick_train_dataset, train_path)
    total_examples = len(pick_train_dataset)
    if is_picktwo_mode:
        print(f"A total of {total_examples} picks in the training set (doubled from {total_examples // 2} PickTwoDraft rows).")
    else:
        print(f"A total of {total_examples} picks in the training set.")
    print(f"Saved training set to {train_path}")
    torch.save(pick_val_dataset, val_path)
    print(f"Saved validation set to {val_path}")

    return train_path, val_path
