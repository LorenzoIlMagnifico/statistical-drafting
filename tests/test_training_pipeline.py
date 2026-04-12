"""
Integration tests for the training pipeline with both model types.

These tests use a tiny synthetic dataset so they run in seconds.

Run with:
    venv/Scripts/python -m pytest tests/test_training_pipeline.py -v
"""
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from statisticaldrafting.model import DraftNet, EmbeddingDraftNet
from statisticaldrafting.train import train_model
from statisticaldrafting.trainingset import PickDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CARDS   = 15
N_TRAIN   = 200
N_VAL     = 40
CARDNAMES = [f"Card_{i}" for i in range(N_CARDS)]
RARITIES  = ["common"] * N_CARDS


def _make_dataset(n):
    """Return a PickDataset with random synthetic examples."""
    rng = np.random.default_rng(42)

    pools     = rng.integers(0, 2, size=(n, N_CARDS)).astype(np.float32)
    packs     = rng.integers(0, 2, size=(n, N_CARDS)).astype(bool)
    positions = rng.random((n, 2)).astype(np.float32)
    missing   = rng.integers(0, 2, size=(n, N_CARDS)).astype(np.float32)

    # pick_vectors: one-hot, always points to a card that is in the pack.
    pick_vectors = np.zeros((n, N_CARDS), dtype=bool)
    for i in range(n):
        on_pack = np.where(packs[i])[0]
        if len(on_pack) == 0:
            packs[i, 0] = True
            on_pack = np.array([0])
        pick_vectors[i, rng.choice(on_pack)] = True

    return PickDataset(pools, packs, pick_vectors, CARDNAMES, RARITIES, positions, missing)


@pytest.fixture(scope="module")
def dataloaders():
    train_ds = _make_dataset(N_TRAIN)
    val_ds   = _make_dataset(N_VAL)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=1,  shuffle=False)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# 1. train_model runs without error on DraftNet (mlp)
# ---------------------------------------------------------------------------

def test_train_mlp_runs(dataloaders, tmp_path):
    train_dl, val_dl = dataloaders
    network = DraftNet(cardnames=CARDNAMES)
    network, info = train_model(
        train_dl, val_dl, network,
        learning_rate=0.01,
        experiment_name="test_mlp",
        model_folder=str(tmp_path) + "/",
    )
    assert info["validation_accuracy"] >= 0.0


# ---------------------------------------------------------------------------
# 2. train_model runs without error on EmbeddingDraftNet (embed)
# ---------------------------------------------------------------------------

def test_train_embed_runs(dataloaders, tmp_path):
    train_dl, val_dl = dataloaders
    network = EmbeddingDraftNet(cardnames=CARDNAMES, embed_dim=16, hidden_dims=(32,))
    network, info = train_model(
        train_dl, val_dl, network,
        learning_rate=0.01,
        experiment_name="test_embed",
        model_folder=str(tmp_path) + "/",
    )
    assert info["validation_accuracy"] >= 0.0


# ---------------------------------------------------------------------------
# 3. training_info dict has all expected keys
# ---------------------------------------------------------------------------

def test_training_info_keys(dataloaders, tmp_path):
    train_dl, val_dl = dataloaders
    network = EmbeddingDraftNet(cardnames=CARDNAMES, embed_dim=8, hidden_dims=(16,))
    _, info = train_model(
        train_dl, val_dl, network,
        learning_rate=0.01,
        experiment_name="test_info",
        model_folder=str(tmp_path) + "/",
    )
    for key in ("experiment_name", "training_picks", "validation_picks",
                "validation_accuracy", "num_epochs", "training_date"):
        assert key in info, f"Missing key '{key}' in training_info"


# ---------------------------------------------------------------------------
# 4. model weights are saved to disk when accuracy improves
#    (force a save by making evaluate_model return 0 on first call)
# ---------------------------------------------------------------------------

def test_model_saved_to_disk(tmp_path):
    import os
    import statisticaldrafting.train as train_mod

    call_count = [0]
    original_eval = train_mod.evaluate_model

    def mock_eval(val_dl, network, device=None):
        call_count[0] += 1
        # Return 0 first (initial baseline), then 50 (improvement triggers save).
        return 0.0 if call_count[0] == 1 else 50.0

    train_mod.evaluate_model = mock_eval
    try:
        train_ds = _make_dataset(64)
        val_ds   = _make_dataset(16)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=1,  shuffle=False)

        network = EmbeddingDraftNet(cardnames=CARDNAMES, embed_dim=8, hidden_dims=(16,))
        train_model(
            train_dl, val_dl, network,
            learning_rate=0.01,
            experiment_name="test_save",
            model_folder=str(tmp_path) + "/",
        )
    finally:
        train_mod.evaluate_model = original_eval

    saved = os.path.join(tmp_path, "test_save.pt")
    assert os.path.exists(saved), f"Model file not found: {saved}"


# ---------------------------------------------------------------------------
# 5. saved weights can be reloaded into a fresh EmbeddingDraftNet
#    Compare two disk-loaded instances — train_model returns the last-epoch
#    network in memory, which may differ from the best-epoch saved weights.
# ---------------------------------------------------------------------------

def test_embed_weights_reload(tmp_path):
    import os
    import statisticaldrafting.train as train_mod

    call_count = [0]
    original_eval = train_mod.evaluate_model

    def mock_eval(val_dl, network, device=None):
        call_count[0] += 1
        return 0.0 if call_count[0] == 1 else 50.0

    train_mod.evaluate_model = mock_eval
    try:
        train_ds = _make_dataset(64)
        val_ds   = _make_dataset(16)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=1,  shuffle=False)

        network = EmbeddingDraftNet(cardnames=CARDNAMES, embed_dim=8, hidden_dims=(16,))
        train_model(
            train_dl, val_dl, network,
            learning_rate=0.01,
            experiment_name="test_reload",
            model_folder=str(tmp_path) + "/",
        )
    finally:
        train_mod.evaluate_model = original_eval

    weights_path = os.path.join(tmp_path, "test_reload.pt")
    assert os.path.exists(weights_path), "No weights file saved"

    # Load the same saved weights into two independent instances and compare.
    state = torch.load(weights_path, weights_only=True)

    model_a = EmbeddingDraftNet(cardnames=CARDNAMES, embed_dim=8, hidden_dims=(16,)).eval()
    model_b = EmbeddingDraftNet(cardnames=CARDNAMES, embed_dim=8, hidden_dims=(16,)).eval()
    model_a.load_state_dict(state)
    model_b.load_state_dict(state)

    pool     = torch.zeros(1, N_CARDS)
    pack     = torch.ones(1, N_CARDS)
    position = torch.zeros(1, 2)
    missing  = torch.zeros(1, N_CARDS)

    with torch.no_grad():
        out_a = model_a(pool, pack, position, missing)
        out_b = model_b(pool, pack, position, missing)

    assert torch.allclose(out_a, out_b), "Two models loaded from the same file produce different output"


# ---------------------------------------------------------------------------
# 6. default_training_pipeline accepts model_type="embed" end-to-end
#    (smoke test — uses cached dataset to avoid file I/O)
# ---------------------------------------------------------------------------

def test_pipeline_model_type_param():
    """
    Verify default_training_pipeline propagates model_type correctly
    by inspecting which network class it instantiates.
    """
    import statisticaldrafting as sd

    created = []
    original_embed = sd.EmbeddingDraftNet
    original_mlp   = sd.DraftNet

    class TrackEmbed(original_embed):
        def __init__(self, *a, **kw):
            created.append("embed")
            super().__init__(*a, **kw)

    class TrackMlp(original_mlp):
        def __init__(self, *a, **kw):
            created.append("mlp")
            super().__init__(*a, **kw)

    sd.EmbeddingDraftNet = TrackEmbed
    sd.DraftNet          = TrackMlp

    try:
        # Patch train_model to bail out immediately after model creation.
        original_train = sd.train_model
        def fake_train(train_dl, val_dl, network, **kw):
            return network, {
                "experiment_name": kw.get("experiment_name", ""),
                "training_picks": 0, "validation_picks": 0,
                "validation_accuracy": 0.0, "num_epochs": 0,
                "training_date": "",
            }
        sd.train_model = fake_train

        # Also patch create_dataset and torch.load to avoid real file I/O.
        import statisticaldrafting.train as train_mod
        import torch

        train_ds = _make_dataset(32)
        val_ds   = _make_dataset(8)

        original_create = train_mod.sd.create_dataset
        train_mod.sd.create_dataset = lambda **kw: ("fake_train.pth", "fake_val.pth")

        original_load = torch.load
        call_count = [0]
        def fake_load(path, **kw):
            call_count[0] += 1
            return train_ds if call_count[0] == 1 else val_ds
        torch.load = fake_load

        import os, unittest.mock as mock
        with mock.patch("os.makedirs"):
            sd.default_training_pipeline(
                "TEST", "Premier",
                overwrite_dataset=False,
                export_onnx=False,
                model_type="embed",
            )

        assert "embed" in created, "EmbeddingDraftNet was not instantiated for model_type='embed'"
        assert "mlp"   not in created, "DraftNet was instantiated instead of EmbeddingDraftNet"

    finally:
        sd.EmbeddingDraftNet = original_embed
        sd.DraftNet          = original_mlp
        sd.train_model       = original_train
        train_mod.sd.create_dataset = original_create
        torch.load           = original_load
