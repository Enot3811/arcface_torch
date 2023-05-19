"""Скрипт для создания embeddings на основе нарезанных окон.
"""


import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.model_tools import load_model, preprocess_model_input
from my_utils.image_tools import read_image


def main(**kwargs):
    win_dir_path: Path = kwargs['cut_windows_path']
    save_path: Path = kwargs['save_path']
    model_path: Path = kwargs['model_path']
    model_name: str = kwargs['model_name']

    img_paths: List[Path] = list(win_dir_path.glob('*'))
    # Отсортировать изображения по номерам классов
    img_paths = list(
        sorted(img_paths, key=lambda path: int(str(path.name)[1:-4])))
    # Прогоняем все окна через модель
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

        model = load_model(model_name, model_path).to(device=device)

        for img_path in tqdm(img_paths):
            window = read_image(img_path)
            window = preprocess_model_input(window).to(device=device)
            embeddings.append(model(window).cpu().numpy())
        
    results = np.concatenate(embeddings, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, results)


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns
    -------
    argparse.Namespace
        Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Создать embeddings из нарезанных окон.'))

    parser.add_argument(
        'cut_windows_path', type=Path,
        help='Путь к директории с нарезанными окнами.')
    parser.add_argument(
        'save_path', type=Path,
        help='Путь для сохранения npy файла с embeddings.')
    parser.add_argument(
        'model_path', type=Path,
        help='Путь к весам обученной модели.')
    parser.add_argument(
        'model_name', type=str,
        help='Название модели.')
    args = parser.parse_args()

    if not args.cut_windows_path.exists():
        raise FileNotFoundError(
            'Указанная директория с нарезанными окнами '
            f'"{args.cut_windows_path}" не существует.')
    if not args.model_path.exists():
        raise FileNotFoundError(
            f'Файл с весами модели "{args.model_path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
