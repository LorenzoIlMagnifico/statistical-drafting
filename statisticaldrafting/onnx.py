import io
import os
import sys

import onnx
import pandas as pd
import statisticaldrafting as sd
import torch

# Note - this code is currently untested in package format due to import issues. 
# It has been successfully tested & run in a notebook. 

def create_onnx_model(model_path, cardnames, onnx_path, model_type="mlp",
                      card_features=None):
    """
    Creates an ONNX model of the network for use in browser code.

    Args:
        model_path    (str): Path to the saved .pt weights file.
        cardnames     (list): Card names for the set.
        onnx_path     (str): Destination .onnx file path.
        model_type    (str): "mlp" for DraftNet, "embed" for EmbeddingDraftNet.
        card_features (torch.Tensor or None): Per-card feature tensor required
            when model_type="embed" and the model was trained with features.
    """
    if model_type == "embed":
        network = sd.EmbeddingDraftNet(cardnames=cardnames,
                                       card_features=card_features)
    else:
        network = sd.DraftNet(cardnames=cardnames)
    network.load_state_dict(torch.load(model_path, weights_only=True))
    network.eval()

    # Dummy inputs (match the forward signature: collection, pack, position, missing)
    dummy_collection = torch.randn(1, len(cardnames)) # [batch, num_cards]
    dummy_pack = torch.ones(1, len(cardnames))        # [batch, num_cards]
    dummy_position = torch.zeros(1, 2)                # [batch, 2]: [pack_number/2, pick_number/14]
    dummy_missing = torch.zeros(1, len(cardnames))    # [batch, num_cards]

    # PyTorch's new ONNX exporter prints emoji progress messages that crash on
    # Windows cp1252 consoles.  Swap stdout to UTF-8 for the duration of export.
    _orig_stdout = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      line_buffering=True)
    except AttributeError:
        pass  # stdout has no .buffer (e.g. redirected); leave as-is

    try:
        torch.onnx.export(
            network,
            (dummy_collection, dummy_pack, dummy_position, dummy_missing),
            onnx_path,
            input_names=["collection", "pack", "position", "missing"],
            output_names=["output"],
            dynamic_axes={
                "collection": {0: "batch_size"},
                "pack": {0: "batch_size"},
                "position": {0: "batch_size"},
                "missing": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=17,  # >=13 for native GELU op
        )
    finally:
        sys.stdout = _orig_stdout

    # PyTorch 2.x can leave a stub on disk; reload + save produces a single complete .onnx.
    onnx.save(onnx.load(onnx_path), onnx_path)
    print(f"Created {onnx_path}")

def create_all_onnx_models(model_dir="../data/models/", onnx_dir="../data/onnx/"):
    """
    Exports all models to ONNX format. 

    Runnable in notebook. 
    """
    model_names = [model_name for model_name in os.listdir(model_dir) if ".pt" in model_name]
    for model_name in model_names:

        # Get cardnames. 
        set = model_name.split("_")[0]
        pick_table = pd.read_csv(
            f"../data/cards/{set}.csv"
        )  # will be sorted.
        cardnames = pick_table["name"].tolist()
        onnx_path = onnx_dir + model_name[:-3] + ".onnx"

        try:
            create_onnx_model(model_dir + model_name, cardnames, onnx_path)
        except:
            print(f"Error for {onnx_path}")