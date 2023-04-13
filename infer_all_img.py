from pathlib import Path
from typing import List
import argparse

import numpy as np
import cv2
import torch
from tqdm import tqdm

from backbones import get_model


def main():
    dset_path = Path(__file__).parents[1] / 'data' / 'img_dataset'
    model_name = 'r50'
    model_path = 'checkpoints/backbone.pth'

    model = load_model(model_name, model_path)

    directories: List[Path] = list(dset_path.glob('*'))
    all_results: List[List[np.ndarray]] = []
    for dir in tqdm(directories):
        images = list(dir.glob('*'))

        class_results: List[np.ndarray] = []
        for image in images:
            class_results.append(infer_img(image, model))
        all_results.append(class_results)
    
    results = np.stack(
        list(map(lambda list_arr: np.concatenate(list_arr), all_results)))
    np.save(dset_path / 'results.npy', results)



def load_model(model_name: str, checkpoint_path: Path) -> torch.nn.Module:
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    return net


def infer_img(img: Path, model: torch.nn.Module) -> np.ndarray:
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return model(img).cpu().detach().numpy()


if __name__ == '__main__':
    main()
