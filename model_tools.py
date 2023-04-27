from pathlib import Path
from typing import Union

import cv2
import numpy as np
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


@torch.no_grad()
def inference(
    model: Union[str, torch.nn.Module],
    name: str,
    img: str
) -> np.ndarray:
    """
    Load a model and an image and then return inference of the model
    on the image.

    Args:
        weight (str): The model or a path to the model checkpoint.
        name (str): A name of the model.
        img (str): A path to the test image.

    Returns:
        np.ndarray: An output of the model.
    """    
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    if isinstance(model, str):
        net = get_model(name, fp16=False)
        net.load_state_dict(torch.load(model))
    else:
        net = model
    net.eval()
    feat = net(img).numpy()
    return feat
