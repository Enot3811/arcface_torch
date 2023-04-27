from pathlib import Path

import torch

from backbones import get_model


def load_model(model_name: str, checkpoint_path: Path) -> torch.nn.Module:
    """
    Load a model from a checkpoint for an inference.

    Args:
        model_name (str): A name of the model.
        checkpoint_path (Path): A path to the checkpoint.

    Returns:
        torch.nn.Module: The loaded model.
    """    
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    return net
