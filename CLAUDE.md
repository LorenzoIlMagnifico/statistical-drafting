# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MTGDraftNet is a machine learning system for Magic: The Gathering draft assistance. It trains neural network models on [17lands](https://www.17lands.com) draft data and provides AI-powered pick recommendations and deck construction. Trained models are exported to ONNX for deployment on [statisticaldrafting.com](http://statisticaldrafting.com).

## Installation & Setup

```bash
pip install -r requirements.txt
# or install as editable package
pip install -e .
```

For the model refresh CI pipeline:
```bash
pip install -r model_refresh/requirements.txt
```

## Common Commands

**Train and export models for all tracked sets:**
```bash
cd model_refresh
python refresh_models.py
```

**Check for new 17lands data (used by CI):**
```bash
cd model_refresh
python ci_check_updates.py
```

**Use the draft assistant in Python:**
```python
from statisticaldrafting import DraftModel

model = DraftModel(set="FDN", draft_mode="Premier")
picks = model.get_pick_order(collection=["Card Name", ...])
deck = model.get_deck_recommendation(pool_cards=[...], starting_colors="WU")
```

**Evaluate deckbuild models:**
```bash
cd deckbuild_sandbox
python run_evaluation.py --set FDN --mode Premier --max-examples 100 --plot
```

## Architecture

Two separate packages handle different tasks:

### `statisticaldrafting/` — Pick Recommendation
- **`model.py`** — `DraftNet`: 2-layer MLP (400 units, GELU, Dropout 0.6, BatchNorm) that scores each card in a pack given the player's current collection. Pack masking zeroes out unavailable cards.
- **`trainingset.py`** — `PickDataset`: Preprocesses 17lands CSV data. Key choices: drops first 7 days of data (stabilization), filters to max-wins games only, encodes cards as one-hot vectors.
- **`train.py`** — Training loop with Adam (lr=0.03), StepLR scheduler (gamma=0.94/epoch), early stopping (40 epochs), and raredraft loss weighting (3× penalty for missing rare picks).
- **`draftassistant.py`** — `DraftModel`: The public inference API. Converts card names to vectors, produces 0–100 ratings (sigmoid-scaled, mean=50), and returns synergy-annotated pick orders.
- **`onnx.py`** — Exports trained `.pt` models to ONNX (opset 17) for browser deployment.

### `statisticaldeckbuild/` — Deck Construction
- **`model.py`** — `DeckbuildNet`: Same MLP shape as DraftNet but predicts the next card to add to a partial deck given the available pool.
- **`deckbuilder.py`** — `IterativeDeckBuilder`: Two-phase construction: (1) mean-field iteration with softmax to refine fractional card counts, then (2) greedy swap of worst deck card vs. best sideboard card until stable.
- **`train.py`** / **`trainingset.py`** — Training and dataset preprocessing for deck construction.

### `model_refresh/` — CI/CD Automation
- **`refresh_models.py`** — Orchestrates full refresh: downloads card lists and draft data from 17lands, trains all models, exports ONNX, updates `data_tracker.json`.
- **`ci_check_updates.py`** — Runs nightly via GitHub Actions (`nightly-training.yml`). Detects new sets on 17lands and creates GitHub Issues to trigger manual refresh.
- `data_tracker.json` — Tracks last-downloaded timestamps to avoid redundant downloads.

### Data Flow

```
17lands CSVs → trainingset.py (preprocessing) → PickDataset / DeckbuildDataset
    → train.py (Adam, CrossEntropy, raredraft weighting) → .pt model
    → onnx.py → .onnx → statisticaldrafting.com frontend
```

## Key Design Decisions

- **Intentionally simple MLP architecture**: 2-layer networks are preferred for maintainability and <1ms CPU inference. Do not introduce complex architectures without strong justification.
- **One-hot card encoding**: Cards are represented as fixed-length binary vectors. The card list for a set is loaded from `data/cards/<SET>.csv`.
- **ONNX export is required for deployment**: All new models must be exportable via `statisticaldrafting/onnx.py` for use in the browser frontend.
- **Model files** (`.pt`) live in GitHub Releases, not in the repo. ONNX files for active sets are committed to `data/onnx/`.
- **Sealed/cube** use the same `DraftModel` API as Premier/Traditional draft modes.
