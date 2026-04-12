import sys
import time
import warnings
import json
import os
from datetime import datetime
from typing import Dict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import statisticaldrafting as sd


def evaluate_model(val_dataloader, network, device=None):
    """
    Evaluate model pick accuracy on validation dataset.
    """
    if device is None:
        device = next(network.parameters()).device
    # Count number correct picks.
    num_correct, num_incorrect = 0, 0
    for pool, pack, human_pick_vector, position, missing in val_dataloader:  # Assumes batch size of 1.
        # TODO: vectorize for performance.
        pool, pack, human_pick_vector, position, missing = pool.to(device), pack.to(device), human_pick_vector.to(device), position.to(device), missing.to(device)
        human_pick_index = torch.argmax(human_pick_vector.int(), 1)
        network.eval()
        with torch.no_grad():
            bot_pick_vector = network(pool.float(), pack.float(), position.float(), missing.float())
            bot_picks_index = torch.argmax(bot_pick_vector, 1)
        if torch.equal(human_pick_index, bot_picks_index):
            num_correct += 1
        else:
            num_incorrect += 1

    # Return and print result.
    percent_correct = 100 * num_correct / (num_correct + num_incorrect)
    print(f"Validation set pick accuracy = {round(percent_correct, 2)}%")
    return percent_correct


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    network: torch.nn.Module,
    learning_rate: float = 0.03,
    experiment_name: str = "test",
    model_folder: str = "../data/models/",
):
    """
    Train and evaluate model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    network = network.to(device)

    # Optimizer parameters.
    # loss_fn = torch.nn.CrossEntropyLoss() # Previous implementation.
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(network.parameters(), lr=learning_rate) # , weight_decay=1e-5)

    # Initial evaluation.
    print(f"Starting to train model. learning_rate={learning_rate}")
    best_percent_correct, best_epoch = evaluate_model(val_dataloader, network, device), 0
    weights_path = model_folder + experiment_name + ".pt"

    # Train model.
    t0 = time.time()
    time_last_message = t0
    epoch = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94) # added
    while (epoch - best_epoch) <= 40:
        network.train()
        epoch_training_loss = list()
        print(f"\nStarting epoch {epoch}  lr={round(scheduler.get_last_lr()[0], 5)}")
        for i, (pool, pack, pick_vector, position, missing) in enumerate(train_dataloader):
            pool, pack, pick_vector, position, missing = pool.to(device), pack.to(device), pick_vector.to(device), position.to(device), missing.to(device)
            optimizer.zero_grad()
            predicted_pick = network(pool.float(), pack.float(), position.float(), missing.float())
            
            # Previous implementation. 
            # loss = loss_fn(predicted_pick, pick_vector.float())
            # loss.backward()

            # print(f"predicted_pick.shape, {predicted_pick.shape}")
            # print(f"pick_vector.float().shape, {pick_vector.float().shape}")

            # New recommended pattern. 
            loss_per_example = loss_fn(predicted_pick, pick_vector.float())
            
            # Find raredraft. 
            rarities = train_dataloader.dataset.rarities
            prediction_rarities = [rarities[i] for i in torch.argmax(predicted_pick, dim=1).tolist()]
            pick_rarities = [rarities[i] for i in torch.argmax(pick_vector.int(), dim=1).tolist()]
            is_raredraft = [(pick in ["common", "uncommon"]) and (pred not in ["common", "uncommon"]) for pick, pred in zip(pick_rarities, prediction_rarities)]
            raredraft_weight = torch.Tensor([3 if rd else 1 for rd in is_raredraft]).to(device) # Raredraft penalty here.
            weighted_loss = loss_per_example * raredraft_weight
            final_loss = weighted_loss.mean()
            final_loss.backward()

            optimizer.step()
            epoch_training_loss.append(final_loss.item())

            # Provide updates every 10 seconds.
            if time_last_message - time.time() > 10:
                examples_processed = (i + 1) * pool.shape[0]
                print(
                    f"Training complete on {examples_processed} examples, time={round(time.time() - t0, 1)}"
                )

        print(f"Training loss: {round(np.mean(epoch_training_loss), 4)}")

        # Evaluate every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            # Evaluation.
            network = network.eval()
            percent_correct = evaluate_model(val_dataloader, network, device)

            # Save best model.
            if percent_correct > best_percent_correct:
                best_percent_correct = percent_correct
                best_epoch = epoch
                weights_path = (
                    model_folder + experiment_name + ".pt"
                )  # TODO: change this
                print(f"Saving model weights to {weights_path}")
                torch.save(network.state_dict(), weights_path)
        epoch += 1
        scheduler.step() # Update learning rate.
        sys.stdout.flush()
    print(f"Training complete for {weights_path}. Best performance={round(best_percent_correct, 2)}% Time={round(time.time()-t0)} seconds\n")
    
    # Return training information dictionary
    training_info = {
        "experiment_name": experiment_name,
        "training_picks": len(train_dataloader.dataset),
        "validation_picks": len(val_dataloader.dataset),
        "validation_accuracy": best_percent_correct,
        "num_epochs": best_epoch,
        "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return network, training_info


def _log_training_info(training_info: dict) -> None:
    """
    Append training information to model_refresh/training_logs.json
    """
    # Determine the path to training_logs.json
    # This function is called from notebooks/ directory, so we need to go up and into model_refresh/
    logs_path = "../model_refresh/training_logs.json"
    
    try:
        # Load existing logs or create empty list
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new training info
        logs.append(training_info)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        
        # Write updated logs
        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Training log saved to {logs_path}")
        
    except Exception as e:
        print(f"Failed to save training log: {e}")


def default_training_pipeline(
    set_abbreviation: str,
    draft_mode: str,
    overwrite_dataset: bool = True,
    export_onnx: bool = True,
    top_drafters_only: bool = False,
    model_type: str = "mlp",
) -> Dict:
    """
    End to end training pipeline using default values.

    Args:
            set_abbreviation (str): Three letter abbreviation of set to create training set of.
            draft_mode (str): Use either "Premier", "Trad", "PickTwo", or "PickTwoTrad" draft data.
            overwrite_dataset (bool): If False, won't overwrite an existing dataset for the set and draft mode.
            export_onnx (bool): If True (default), export a browser-ready ONNX model alongside the .pt weights.
            model_type (str): "mlp" for the original DraftNet, "embed" for EmbeddingDraftNet.
    """
    import sys
    print("Step 1: Creating dataset...")
    sys.stdout.flush()

    # Build model name suffix from flags.
    model_suffix = ""
    if top_drafters_only:
        model_suffix += "_top3pct"
    if model_type == "embed":
        model_suffix += "_embed"
    train_path, val_path = sd.create_dataset(
        set_abbreviation=set_abbreviation,
        draft_mode=draft_mode,
        overwrite=overwrite_dataset,
        top_drafters_only=top_drafters_only,
    )

    print(f"Step 2: Loading datasets from {train_path} and {val_path}...")
    sys.stdout.flush()

    dataset_folder = "../data/training_sets/"

    train_dataset = torch.load(train_path, weights_only=False)

    batch_size = 1000
    learning_rate = 0.003

    print(f"Step 3: Creating train dataloader (batch_size={batch_size})...")
    sys.stdout.flush()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Step 4: Loading validation dataset...")
    sys.stdout.flush()

    val_dataset = torch.load(val_path, weights_only=False)
    print("Step 5: Creating validation dataloader...")
    sys.stdout.flush()

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Step 6: Creating model (type={model_type})...")
    sys.stdout.flush()

    card_features = None
    if model_type == "embed":
        from statisticaldrafting.card_features import load_card_features
        card_features = load_card_features(
            set_abbrev=set_abbreviation,
            draft_mode=draft_mode,
            cardnames=train_dataset.cardnames,
        )
        if card_features is not None:
            print(f"Loaded card features: {card_features.shape[1]} features per card.")
        network = sd.EmbeddingDraftNet(
            cardnames=train_dataset.cardnames,
            card_features=card_features,
        )
    elif model_type == "mlp":
        network = sd.DraftNet(cardnames=train_dataset.cardnames)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'mlp' or 'embed'.")

    print(f"Step 7: Starting model training (lr={learning_rate})...")
    sys.stdout.flush()

    experiment_name = f"{set_abbreviation}_{draft_mode}{model_suffix}"
    network, training_info = sd.train_model(
        train_dataloader,
        val_dataloader,
        network,
        learning_rate=learning_rate,
        experiment_name=experiment_name,
    )

    model_path = f"../data/models/{experiment_name}.pt"
    onnx_path = f"../data/onnx/{experiment_name}.onnx"

    if export_onnx:
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        print(f"Exporting model to ONNX format: {onnx_path}")
        onnx_card_features = card_features if model_type == "embed" else None
        sd.create_onnx_model(
            model_path=model_path,
            cardnames=train_dataset.cardnames,
            onnx_path=onnx_path,
            model_type=model_type,
            card_features=onnx_card_features,
        )
    
    # Log training information to training_logs.json
    _log_training_info(training_info)
    
    return training_info 