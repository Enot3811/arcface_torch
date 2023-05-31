"""Скрипт для предсказания пачки семплов.

Тестирование проводится на основе сохранённых заранее npy файлов с embeddings.
"""


import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import normalize, angular_many2many
from my_utils.image_tools import read_image


def main(**kwargs):
    dset_emb_path = kwargs['dataset_embeddings']
    test_emb_path = kwargs['test_embeddings']
    cls_names_path = kwargs['cls_names']
    cls_dir = kwargs['classes_images']
    samples_dir = kwargs['samples_images']
    show_images = kwargs['show_images']

    smp_img_paths: List[Path] = list(samples_dir.glob('*'))
    # Отсортировать изображения по номерам классов
    smp_img_paths = list(
        sorted(smp_img_paths, key=lambda path: int(str(path.name)[1:-4])))
    
    cls_img_paths: List[Path] = list(cls_dir.glob('*'))
    cls_img_paths = filter(lambda path: not path.is_dir(), cls_img_paths)
    # Отсортировать изображения по номерам классов
    cls_img_paths = list(
        sorted(cls_img_paths, key=lambda path: int(path.name[1:-4])))

    # Читаем файл с названиями классов и соответствующими им индексами
    with open(cls_names_path) as f:
        idx_to_name = {}
        for line in f.readlines():
            idx, name = line.split()
            idx_to_name[int(idx)] = name

    # (n_cls, n_dset_samples, embed) или (n_cls, 1, embed)
    dset_emb: np.ndarray = np.load(dset_emb_path)
    # (n_samples, embed)
    test_emb: np.ndarray = np.load(test_emb_path)

    # Нормализуем
    dset_emb = normalize(dset_emb)
    test_emb = normalize(test_emb)

    # Вычисляем центроиды классов датасета, с которыми будем сравниваться
    dset_centroids = dset_emb.mean(axis=1)
    
    # Классификация
    n_samples = test_emb.shape[0]
    n_cls = dset_centroids.shape[0]
    angles = angular_many2many(test_emb, dset_centroids)
    
    for i in range(angles.shape[0]):
        nearest = np.argsort(angles[i])
        print(f'Изображение {i}:')
        print('Индексы ближайших классов:', nearest[:3])
        names = [idx_to_name[idx] for idx in nearest[:3]]
        print('Имена ближайших классов:', names)
        print()

        # Отображение текущего изображения и ближайших к нему.
        if show_images:
            fig, axes = plt.subplots(1, 4, figsize=(40, 40))
            samples_img = read_image(smp_img_paths[i])
            axes[0].imshow(samples_img)
            for j, name in enumerate(names):
                cls_img = read_image(cls_dir / (name + '.png'))
                axes[j + 1].imshow(cls_img)
            plt.show()


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Протестировать модель. '
                     'Посчитать метрики на основе выходных данных модели.'))

    parser.add_argument(
        'dataset_embeddings', type=Path,
        help='Путь к файлу с embeddings от датасета.')
    parser.add_argument(
        'test_embeddings', type=Path,
        help='Путь к файлу с тестируемыми embeddings.')
    parser.add_argument(
        'cls_names', type=Path,
        help='Путь к файлу с соотношениями индексов и названий классов.')
    parser.add_argument(
        'classes_images', type=Path,
        help='Путь к директории с классовыми изображениями.')
    parser.add_argument(
        'samples_images', type=Path,
        help='Путь к директории с проверяемыми изображениями.')
    parser.add_argument(
        '--show_images', action='store_true',
        help=('Отобразить классифицируемое изображение и изображения '
              'ближайших классов'))
    args = parser.parse_args()

    for path in {args.dataset_embeddings, args.test_embeddings}:
        if not path.exists():
            raise FileNotFoundError(
                f'Указанный файл с embeddings "{path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
