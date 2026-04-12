"""
Tests for EmbeddingDraftNet and card_features.

Run with:
    venv/Scripts/python -m pytest tests/test_embedding_model.py -v
"""
import pytest
import torch
import torch.nn as nn

from statisticaldrafting.model import EmbeddingDraftNet
from statisticaldrafting.card_features import (
    load_card_features, CARD_FEATURE_DIM, _card_feature_vector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CARDS = 20   # small synthetic set
BATCH   = 4
EMBED   = 16   # small embed dim so tests run fast


@pytest.fixture
def cardnames():
    return [f"Card_{i}" for i in range(N_CARDS)]


@pytest.fixture
def model(cardnames):
    return EmbeddingDraftNet(cardnames, embed_dim=EMBED, hidden_dims=(32, 16))


def _dummy_batch(n_cards=N_CARDS, batch=BATCH, n_pool=5, n_pack=8):
    """Return (pool, pack, position, missing) tensors."""
    pool     = torch.zeros(batch, n_cards)
    pack     = torch.zeros(batch, n_cards)
    missing  = torch.zeros(batch, n_cards)
    position = torch.rand(batch, 2)

    for b in range(batch):
        pool_idx = torch.randperm(n_cards)[:n_pool]
        pack_idx = torch.randperm(n_cards)[:n_pack]
        miss_idx = torch.randperm(n_cards)[:3]
        pool[b, pool_idx] = 1.0
        pack[b, pack_idx] = 1.0
        missing[b, miss_idx] = 1.0

    return pool, pack, position, missing


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

def test_output_shape(model):
    pool, pack, position, missing = _dummy_batch()
    out = model(pool, pack, position, missing)
    assert out.shape == (BATCH, N_CARDS), (
        f"Expected ({BATCH}, {N_CARDS}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Pack masking — scores outside the pack must be exactly zero
# ---------------------------------------------------------------------------

def test_pack_masking(model):
    pool, pack, position, missing = _dummy_batch()
    out = model(pool, pack, position, missing)

    # Wherever pack is 0, output must be 0.
    off_pack = (pack == 0)
    assert (out[off_pack] == 0.0).all(), "Non-zero scores for cards outside the pack"


# ---------------------------------------------------------------------------
# 3. Empty pool — no NaN/Inf when pool is all-zeros (first pick of the draft)
# ---------------------------------------------------------------------------

def test_empty_pool_no_nan(model):
    pool     = torch.zeros(BATCH, N_CARDS)
    pack     = torch.ones(BATCH, N_CARDS)
    position = torch.zeros(BATCH, 2)
    missing  = torch.zeros(BATCH, N_CARDS)

    out = model(pool, pack, position, missing)
    assert not torch.isnan(out).any(), "NaN in output with empty pool"
    assert not torch.isinf(out).any(), "Inf in output with empty pool"


# ---------------------------------------------------------------------------
# 4. Empty missing — no NaN/Inf when missing is all-zeros
# ---------------------------------------------------------------------------

def test_empty_missing_no_nan(model):
    pool, pack, position, _ = _dummy_batch()
    missing = torch.zeros(BATCH, N_CARDS)

    out = model(pool, pack, position, missing)
    assert not torch.isnan(out).any(), "NaN in output with empty missing"
    assert not torch.isinf(out).any(), "Inf in output with empty missing"


# ---------------------------------------------------------------------------
# 5. Determinism — same input gives same output in eval mode
# ---------------------------------------------------------------------------

def test_determinism(model):
    model.eval()
    pool, pack, position, missing = _dummy_batch()
    with torch.no_grad():
        out1 = model(pool, pack, position, missing)
        out2 = model(pool, pack, position, missing)
    assert torch.equal(out1, out2), "Model output is not deterministic in eval mode"


# ---------------------------------------------------------------------------
# 6. Different pools produce different scores (model is not degenerate)
# ---------------------------------------------------------------------------

def test_pool_affects_scores(model):
    model.eval()
    _, pack, position, missing = _dummy_batch(batch=1)

    pool_a = torch.zeros(1, N_CARDS)
    pool_b = torch.ones(1, N_CARDS)

    with torch.no_grad():
        out_a = model(pool_a, pack, position, missing)
        out_b = model(pool_b, pack, position, missing)

    assert not torch.equal(out_a, out_b), (
        "Scores identical for empty vs full pool — pool context has no effect"
    )


# ---------------------------------------------------------------------------
# 7. Gradients flow through the embedding table during training
# ---------------------------------------------------------------------------

def test_gradients_flow(model):
    pool, pack, position, missing = _dummy_batch()
    target = torch.zeros(BATCH, N_CARDS)
    for b in range(BATCH):
        on_pack = pack[b].nonzero(as_tuple=True)[0]
        target[b, on_pack[0]] = 1.0   # pick first available card

    loss_fn = nn.CrossEntropyLoss()
    out = model(pool, pack, position, missing)
    loss = loss_fn(out, target)
    loss.backward()

    embed_grad = model.card_embed.weight.grad
    assert embed_grad is not None, "No gradient on embedding table"
    assert embed_grad.abs().sum() > 0, "Embedding gradient is all zeros"


# ---------------------------------------------------------------------------
# 8. Batch size of 1 works (edge case for val loop which uses batch_size=1)
# ---------------------------------------------------------------------------

def test_batch_size_one(model):
    pool, pack, position, missing = _dummy_batch(batch=1)
    out = model(pool, pack, position, missing)
    assert out.shape == (1, N_CARDS)


# ---------------------------------------------------------------------------
# 9. embed_dim and hidden_dims are respected
# ---------------------------------------------------------------------------

def test_architecture_params(cardnames):
    m = EmbeddingDraftNet(cardnames, embed_dim=32, hidden_dims=(64, 32, 16))
    assert m.card_embed.embedding_dim == 32
    # score_head: Linear, GELU, Dropout (x3 layers) + final Linear = 10 modules
    linears = [mod for mod in m.score_head if isinstance(mod, nn.Linear)]
    assert len(linears) == 4   # 3 hidden + 1 output


# ---------------------------------------------------------------------------
# 10. cardnames attribute is preserved
# ---------------------------------------------------------------------------

def test_cardnames_stored(model, cardnames):
    assert model.cardnames == cardnames


# ---------------------------------------------------------------------------
# 11. card_features: feature vector has correct dimension and value ranges
# ---------------------------------------------------------------------------

def test_card_feature_vector_dim():
    card = {
        "name": "Test Card",
        "cmc": 3,
        "colors": ["W", "U"],
        "types": ["Creature", "Enchantment"],
        "rarity": "rare",
        "deck_colors": {
            "All Decks": {"gihwr": 55.0, "ohwr": 52.0, "alsa": 3.0, "ata": 4.0, "iwd": 1.5}
        },
    }
    vec = _card_feature_vector(card)
    assert len(vec) == CARD_FEATURE_DIM, f"Expected {CARD_FEATURE_DIM} features, got {len(vec)}"

    # cmc=3 -> 0.3
    assert abs(vec[0] - 0.3) < 1e-5
    # W=1, U=1, B=0, R=0, G=0
    assert vec[1:6] == [1.0, 1.0, 0.0, 0.0, 0.0]
    # Creature=1, Instant=0, Sorcery=0, Enchantment=1, ...
    assert vec[6] == 1.0  # Creature
    assert vec[9] == 1.0  # Enchantment
    # rarity rare -> index 15 = 1
    assert vec[15] == 1.0  # rare
    assert vec[13] == 0.0  # not common


# ---------------------------------------------------------------------------
# 12. EmbeddingDraftNet with card_features: output shape unchanged
# ---------------------------------------------------------------------------

def test_model_with_card_features(cardnames):
    feats = torch.randn(N_CARDS, CARD_FEATURE_DIM)
    m = EmbeddingDraftNet(cardnames, embed_dim=EMBED, hidden_dims=(32,), card_features=feats)

    pool, pack, position, missing = _dummy_batch()
    out = m(pool, pack, position, missing)
    assert out.shape == (BATCH, N_CARDS)
    # Pack masking still holds
    assert (out[pack == 0] == 0.0).all()


# ---------------------------------------------------------------------------
# 13. card_features buffer moves to the same device as the model
# ---------------------------------------------------------------------------

def test_card_features_device(cardnames):
    feats = torch.randn(N_CARDS, CARD_FEATURE_DIM)
    m = EmbeddingDraftNet(cardnames, embed_dim=EMBED, hidden_dims=(16,), card_features=feats)
    # Move to CPU explicitly (always available in test environment).
    m = m.to("cpu")
    assert m.card_features.device.type == "cpu"


# ---------------------------------------------------------------------------
# 14. load_card_features returns None when no file exists
# ---------------------------------------------------------------------------

def test_load_card_features_missing(cardnames, tmp_path):
    result = load_card_features("FAKE", "Premier", cardnames, data_dir=str(tmp_path))
    assert result is None


# ---------------------------------------------------------------------------
# 15. load_card_features returns correct tensor for a synthetic JSON
# ---------------------------------------------------------------------------

def test_load_card_features_synthetic(cardnames, tmp_path):
    import json

    card_ratings = {}
    for i, name in enumerate(cardnames):
        card_ratings[str(i)] = {
            "name": name,
            "cmc": i % 8,
            "colors": ["W"] if i % 2 == 0 else ["B"],
            "types": ["Creature"],
            "rarity": "common",
            "isprimarycard": 1,
            "deck_colors": {
                "All Decks": {"gihwr": 50.0, "ohwr": 50.0, "alsa": 5.0, "ata": 5.0, "iwd": 0.0}
            },
        }

    payload = {"card_ratings": card_ratings}
    path = tmp_path / "SYN_PremierDraft_Data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    tensor = load_card_features("SYN", "Premier", cardnames, data_dir=str(tmp_path))
    assert tensor is not None
    assert tensor.shape == (N_CARDS, CARD_FEATURE_DIM)
    assert tensor.dtype == torch.float32
