#!/usr/bin/env python3
"""
Train an enhanced DraftNet on enriched FDN datasets and compare to the baseline.

The enriched dataset adds per-card features (GIH WR, CMC, colors) and archetype
win rates. EnhancedDraftNet aggregates card features over the pool via a dot
product and appends archetype win rates to the baseline input vector.

Usage:
    python train_enhanced.py FDN
    python train_enhanced.py FDN Premier --no-onnx
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from statisticaldrafting.trainingset import EnhancedPickDataset  # noqa: F401 — needed for pickle


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EnhancedDraftNet(nn.Module):
    """
    Same 2-layer MLP as DraftNet, extended with two extra input signals:

    pool_agg (7):      dot product of pool one-hot with card_features matrix,
                       giving aggregate GIH WR / CMC / color counts for held cards
    archetype_wr (10): two-color pair win rates for the current set

    Total input dim: 2*N + 2 + 7 + 10  (vs. 2*N + 2 for baseline)
    """

    N_CARD_FEATURES = 7
    N_ARCHETYPE = 10

    def __init__(self, cardnames):
        super().__init__()
        self.cardnames = cardnames
        N = len(cardnames)
        input_dim = N * 2 + 2 + self.N_CARD_FEATURES + self.N_ARCHETYPE
        hidden_dims = [input_dim, 400, 400]

        self.dropout = nn.Dropout(0.6)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in hidden_dims[1:])
        self.output_layer = nn.Linear(hidden_dims[-1], N)

    def forward(self, pool, pack, position, missing, card_features, archetype_wr):
        # pool: (B, N)  card_features: (B, N, 7) -> pool_agg: (B, 7)
        pool_agg = torch.bmm(pool.float().unsqueeze(1), card_features.float()).squeeze(1)

        x = torch.cat([
            pool.float(),
            missing.float(),
            position.float(),
            pool_agg,
            archetype_wr.float(),
        ], dim=-1)

        for layer, norm in zip(self.hidden_layers, self.norms):
            x = layer(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = norm(x)

        x = self.output_layer(x)
        x = x * pack.float()
        return x


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def evaluate(val_dataloader, network, device):
    network.eval()
    num_correct = num_incorrect = 0
    with torch.no_grad():
        for pool, pack, pick_vec, position, missing, card_features, archetype_wr in val_dataloader:
            pool, pack, pick_vec, position, missing, card_features, archetype_wr = (
                t.to(device) for t in (pool, pack, pick_vec, position, missing, card_features, archetype_wr)
            )
            human_pick = torch.argmax(pick_vec.int(), 1)
            scores = network(pool, pack, position, missing, card_features, archetype_wr)
            bot_pick = torch.argmax(scores, 1)
            num_correct   += (human_pick == bot_pick).sum().item()
            num_incorrect += (human_pick != bot_pick).sum().item()

    acc = 100 * num_correct / (num_correct + num_incorrect)
    print(f"Validation accuracy = {round(acc, 2)}%")
    return acc


def train(train_dataloader, val_dataloader, network, learning_rate, experiment_name, model_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    network = network.to(device)

    loss_fn  = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    # Cosine decay from learning_rate to eta_min over T_max epochs, then restarts.
    # Warm restarts every 30 epochs keep the LR from dying to zero, which caused
    # the model to stall in the StepLR run.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, eta_min=1e-5
    )

    best_acc, best_epoch = evaluate(val_dataloader, network, device), 0
    weights_path = os.path.join(model_folder, experiment_name + ".pt")

    t0 = time.time()
    epoch = 0
    while (epoch - best_epoch) <= 40:
        network.train()
        epoch_loss = []
        print(f"\nStarting epoch {epoch}  lr={round(optimizer.param_groups[0]['lr'], 6)}")

        for pool, pack, pick_vec, position, missing, card_features, archetype_wr in train_dataloader:
            pool, pack, pick_vec, position, missing, card_features, archetype_wr = (
                t.to(device) for t in (pool, pack, pick_vec, position, missing, card_features, archetype_wr)
            )
            optimizer.zero_grad()
            scores = network(pool, pack, position, missing, card_features, archetype_wr)

            loss_per = loss_fn(scores, pick_vec.float())

            rarities = train_dataloader.dataset.rarities
            pred_rarities = [rarities[i] for i in torch.argmax(scores, 1).tolist()]
            pick_rarities = [rarities[i] for i in torch.argmax(pick_vec.int(), 1).tolist()]
            is_raredraft = [
                (pick in ("common", "uncommon")) and (pred not in ("common", "uncommon"))
                for pick, pred in zip(pick_rarities, pred_rarities)
            ]
            weight = torch.tensor([3.0 if rd else 1.0 for rd in is_raredraft], device=device)
            (loss_per * weight).mean().backward()

            optimizer.step()
            epoch_loss.append((loss_per * weight).mean().item())

        print(f"Training loss: {round(np.mean(epoch_loss), 4)}")

        if epoch % 2 == 0 and epoch > 0:
            acc = evaluate(val_dataloader, network, device)
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                torch.save(network.state_dict(), weights_path)
                print(f"Saved model -> {weights_path}")

        epoch += 1
        scheduler.step(epoch)
        sys.stdout.flush()

    print(f"\nTraining complete. Best accuracy={round(best_acc, 2)}%  epoch={best_epoch}  time={round(time.time()-t0)}s")
    return network, best_acc, best_epoch


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_models(baseline_path, enhanced_path, val_dataloader, cardnames, device):
    """Load both saved models and report their validation accuracy side by side."""
    print("\n=== Model Comparison ===")

    # Baseline
    sys.path.insert(0, SCRIPT_DIR)
    import statisticaldrafting as sd

    baseline = sd.DraftNet(cardnames=cardnames).to(device)
    baseline.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    baseline.eval()

    base_correct = base_total = 0
    with torch.no_grad():
        for pool, pack, pick_vec, position, missing, card_features, archetype_wr in val_dataloader:
            pool, pack, pick_vec, position, missing = (
                t.to(device) for t in (pool, pack, pick_vec, position, missing)
            )
            human_pick = torch.argmax(pick_vec.int().to(device), 1)
            scores = baseline(pool, pack, position, missing)
            base_correct += (torch.argmax(scores, 1) == human_pick).sum().item()
            base_total   += len(human_pick)
    base_acc = 100 * base_correct / base_total

    # Enhanced
    enhanced = EnhancedDraftNet(cardnames=cardnames).to(device)
    enhanced.load_state_dict(torch.load(enhanced_path, map_location=device, weights_only=True))
    enhanced.eval()

    enh_correct = enh_total = 0
    with torch.no_grad():
        for pool, pack, pick_vec, position, missing, card_features, archetype_wr in val_dataloader:
            pool, pack, pick_vec, position, missing, card_features, archetype_wr = (
                t.to(device) for t in (pool, pack, pick_vec, position, missing, card_features, archetype_wr)
            )
            human_pick = torch.argmax(pick_vec.int(), 1)
            scores = enhanced(pool, pack, position, missing, card_features, archetype_wr)
            enh_correct += (torch.argmax(scores, 1) == human_pick).sum().item()
            enh_total   += len(human_pick)
    enh_acc = 100 * enh_correct / enh_total

    print(f"  Baseline (DraftNet)          : {round(base_acc, 2)}%")
    print(f"  Enhanced (EnhancedDraftNet)  : {round(enh_acc, 2)}%")
    print(f"  Delta                        : {round(enh_acc - base_acc, 2):+}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_code", help="Set abbreviation (e.g. FDN)")
    parser.add_argument("draft_mode", nargs="?", default="Premier")
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    set_code  = args.set_code.upper()
    draft_mode = args.draft_mode

    training_sets_dir = os.path.join(SCRIPT_DIR, "data", "training_sets")
    models_dir        = os.path.join(SCRIPT_DIR, "data", "models")
    train_path = os.path.join(training_sets_dir, f"{set_code}_{draft_mode}_train_enriched.pth")
    val_path   = os.path.join(training_sets_dir, f"{set_code}_{draft_mode}_val_enriched.pth")

    for p in (train_path, val_path):
        if not os.path.exists(p):
            print(f"ERROR: Enriched dataset not found: {p}")
            print("Run enrich_dataset.py first.")
            sys.exit(1)

    print(f"=== Enhanced Training: {set_code} {draft_mode} ===\n")
    print("Loading datasets...")
    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset   = torch.load(val_path,   weights_only=False)
    print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Cards: {len(train_dataset.cardnames)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    network = EnhancedDraftNet(cardnames=train_dataset.cardnames)
    experiment_name = f"{set_code}_{draft_mode}_enhanced"

    network, best_acc, best_epoch = train(
        train_loader, val_loader, network,
        learning_rate=args.learning_rate,
        experiment_name=experiment_name,
        model_folder=models_dir,
    )

    enhanced_path = os.path.join(models_dir, experiment_name + ".pt")
    baseline_path = os.path.join(models_dir, f"{set_code}_{draft_mode}.pt")

    if os.path.exists(baseline_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compare_models(baseline_path, enhanced_path, val_loader, train_dataset.cardnames, device)
    else:
        print(f"\nBaseline model not found at {baseline_path}, skipping comparison.")


if __name__ == "__main__":
    main()
