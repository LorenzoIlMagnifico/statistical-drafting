"""
predict_deck_wins.py

Trains an MLP that predicts event_match_wins (float) from a deck's card composition.

Usage
-----
    python predict_deck_wins.py \
        --csv  ../data/17-lands-data/draft_data_public.TMT.PremierDraft.csv \
        --cards ../data/cards/TMT.csv \
        --save  ../data/deck_wins/TMT_Premier_decks.npz \
        --model ../data/deck_wins/TMT_Premier_wins_model.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from deck_dataset import load_deck_dataset, save_deck_dataset, load_saved_deck_dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeckWinsDataset(Dataset):
    def __init__(self, deck_matrix: np.ndarray, wins: np.ndarray):
        self.decks = torch.from_numpy(deck_matrix).float()
        self.wins  = torch.from_numpy(wins).float()

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, idx):
        return self.decks[idx], self.wins[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeckWinsNet(nn.Module):
    """
    2-layer MLP that predicts the number of wins for a given deck.

    Input  : one-hot / count vector of cards in the maindeck  (N_cards,)
    Output : scalar win prediction  (float)
    """

    def __init__(self, n_cards: int, hidden_dim: int = 400, dropout: float = 0.4):
        super().__init__()
        self.fc1   = nn.Linear(n_cards, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.out   = nn.Linear(hidden_dim, 1)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.bn1(self.act(self.fc1(x))))
        x = self.drop(self.bn2(self.act(self.fc2(x))))
        return self.out(x).squeeze(-1)          # (B,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for decks, wins in loader:
            decks, wins = decks.to(device), wins.to(device)
            preds = model(decks)
            total_loss += loss_fn(preds, wins).item() * len(wins)
            n += len(wins)
    return total_loss / n   # mean MSE


def train(
    dataset: DeckWinsDataset,
    n_cards: int,
    model_path: str,
    hidden_dim: int = 400,
    dropout: float = 0.4,
    learning_rate: float = 0.001,
    batch_size: int = 512,
    train_fraction: float = 0.8,
    patience: int = 40,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train / val split
    n_train = int(len(dataset) * train_fraction)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train: {n_train:,} drafts   Val: {n_val:,} drafts")

    model = DeckWinsNet(n_cards=n_cards, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch    = 0
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    t0 = time.time()

    epoch = 0
    while (epoch - best_epoch) <= patience:
        model.train()
        train_losses = []
        for decks, wins in train_loader:
            decks, wins = decks.to(device), wins.to(device)
            optimizer.zero_grad()
            preds = model(decks)
            loss  = loss_fn(preds, wins)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_mse  = evaluate(model, val_loader, device)
        val_rmse = val_mse ** 0.5
        print(
            f"Epoch {epoch:3d}  "
            f"lr={scheduler.get_last_lr()[0]:.5f}  "
            f"train_loss={np.mean(train_losses):.4f}  "
            f"val_rmse={val_rmse:.4f}"
        )

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch    = epoch
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (val_rmse={val_rmse:.4f})")

        scheduler.step()
        epoch += 1
        sys.stdout.flush()

    elapsed = round(time.time() - t0)
    print(
        f"\nTraining complete. Best epoch={best_epoch}  "
        f"best_val_rmse={best_val_loss**0.5:.4f}  "
        f"time={elapsed}s"
    )
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deck-wins prediction model.")
    parser.add_argument("--csv",   required=False,
                        default="../data/17-lands-data/draft_data_public.TMT.PremierDraft.csv",
                        help="Path to 17lands draft CSV")
    parser.add_argument("--cards", required=False,
                        default="../data/cards/TMT.csv",
                        help="Path to set card list CSV")
    parser.add_argument("--save",  required=False,
                        default="../data/deck_wins/TMT_Premier_decks.npz",
                        help="Where to save/load the extracted deck dataset (.npz)")
    parser.add_argument("--model", required=False,
                        default="../data/deck_wins/TMT_Premier_wins_model.pt",
                        help="Where to save the trained model weights (.pt)")
    args = parser.parse_args()

    # Load or extract deck dataset
    if os.path.exists(args.save):
        deck_matrix, wins, card_names = load_saved_deck_dataset(args.save)
    else:
        deck_matrix, wins, card_names = load_deck_dataset(
            csv_path=args.csv,
            cards_path=args.cards if os.path.exists(args.cards) else None,
        )
        save_deck_dataset(args.save, deck_matrix, wins, card_names)

    dataset = DeckWinsDataset(deck_matrix, wins)
    train(dataset, n_cards=len(card_names), model_path=args.model)
