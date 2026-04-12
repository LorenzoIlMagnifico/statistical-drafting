"""
Optuna hyperparameter search for EmbeddingDraftNet.

Uses the top3pct dataset for fast iteration. Each trial trains with a
reduced early-stopping patience so it finishes in a few minutes.

Usage (from project root):
    python tune_embedding.py FDN Premier [--n-trials 50] [--study-name fdn_embed]

Results are persisted to a local SQLite DB so the study can be resumed:
    python tune_embedding.py FDN Premier --n-trials 20   # first run
    python tune_embedding.py FDN Premier --n-trials 30   # resume, adds 30 more
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.chdir(os.path.join(os.path.dirname(__file__), "notebooks"))

import torch
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner

import statisticaldrafting as sd
from statisticaldrafting.model import EmbeddingDraftNet
from statisticaldrafting.train import evaluate_model


# ---------------------------------------------------------------------------
# Training loop for a single trial (stripped-down version of train_model)
# ---------------------------------------------------------------------------

def train_trial(trial, train_dl, val_dl, n_cards, patience=10):
    """
    Train EmbeddingDraftNet with hyperparams suggested by Optuna.
    Reports intermediate accuracy for pruning and returns best accuracy.
    """
    # --- Hyperparameter search space ---
    embed_dim   = trial.suggest_categorical("embed_dim", [32, 64, 128, 256])
    n_layers    = trial.suggest_int("n_layers", 1, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    dropout     = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [512, 1000, 2000])

    hidden_dims = tuple([hidden_size] * n_layers)
    cardnames   = train_dl.dataset.cardnames

    # Rebuild dataloader if batch size changed (trial-specific).
    trial_train_dl = DataLoader(
        train_dl.dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = EmbeddingDraftNet(
        cardnames=cardnames, embed_dim=embed_dim, hidden_dims=hidden_dims
    ).to(device)

    # Replace dropout in score_head with trial value.
    for module in network.score_head:
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    loss_fn   = torch.nn.CrossEntropyLoss()

    best_acc, epochs_without_improvement = 0.0, 0
    epoch = 0

    while epochs_without_improvement <= patience:
        network.train()
        for pool, pack, pick_vec, position, missing in trial_train_dl:
            pool, pack, pick_vec, position, missing = (
                pool.to(device), pack.to(device), pick_vec.to(device),
                position.to(device), missing.to(device),
            )
            optimizer.zero_grad()
            out  = network(pool.float(), pack.float(), position.float(), missing.float())
            loss = loss_fn(out, pick_vec.float())
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate every 2 epochs.
        if epoch % 2 == 0:
            acc = evaluate_model(val_dl, network, device)
            trial.report(acc, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if acc > best_acc:
                best_acc = acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 2

        epoch += 1

    return best_acc


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(train_dl, val_dl, n_cards):
    def objective(trial):
        return train_trial(trial, train_dl, val_dl, n_cards)
    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set",  help="Set abbreviation, e.g. FDN")
    parser.add_argument("mode", help="Draft mode, e.g. Premier")
    parser.add_argument("--n-trials",   type=int, default=30,
                        help="Number of Optuna trials to run (default: 30)")
    parser.add_argument("--study-name", default=None,
                        help="Study name for resuming (default: <set>_<mode>_embed)")
    parser.add_argument("--patience",   type=int, default=10,
                        help="Early stopping patience per trial in epochs (default: 10)")
    args = parser.parse_args()

    study_name = args.study_name or f"{args.set}_{args.mode}_embed"
    db_path    = f"../data/optuna_{study_name}.db"
    storage    = f"sqlite:///{db_path}"

    # Load top3pct dataset.
    dataset_suffix = "_top3pct"
    train_path = f"../data/training_sets/{args.set}_{args.mode}{dataset_suffix}_train.pth"
    val_path   = f"../data/training_sets/{args.set}_{args.mode}{dataset_suffix}_val.pth"

    if not os.path.exists(train_path):
        print(f"Top3pct dataset not found at {train_path}")
        print("Run first: python run_training.py <SET> <MODE> --top3 --keep-dataset")
        sys.exit(1)

    print(f"Loading datasets from {train_path} ...")
    train_ds = torch.load(train_path, weights_only=False)
    val_ds   = torch.load(val_path,   weights_only=False)
    train_dl = DataLoader(train_ds, batch_size=1000, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=1,    shuffle=False, num_workers=0)
    n_cards  = len(train_ds.cardnames)

    print(f"Study: {study_name}  |  DB: {db_path}  |  Trials: {args.n_trials}")
    print(f"Cards: {n_cards}  |  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=6),
        load_if_exists=True,
    )

    study.optimize(
        make_objective(train_dl, val_dl, n_cards),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n===== Best trial =====")
    best = study.best_trial
    print(f"  Accuracy : {best.value:.2f}%")
    print(f"  Params   :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\nTo train final model with best params, run:")
    print(f"  (implement train_best.py or adapt run_training.py)")


if __name__ == "__main__":
    main()
