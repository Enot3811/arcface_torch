"""Скрипт для тестирования модели.

Тестирование проводится на основе сохранённых заранее npy файлов с embeddings.
"""


import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import normalize, angular_many2many, calculate_accuracy
from my_utils.grid_tools import create_overlapping_heatmap, Colormap


def main(**kwargs):
    dset_emb_path: Path = kwargs['dataset_embeddings']
    test_emb_path: Path = kwargs['test_embeddings']
    show_progress: bool = kwargs['show_calc_progress']
    map_size: Optional[Tuple[int, int]] = kwargs['map_h_w']
    step: Optional[int] = kwargs['step']
    win_size: Optional[int] = kwargs['win_size']

    dset_emb: np.ndarray = np.load(dset_emb_path)  # (n_cls, n_samples, embed)
    test_emb: np.ndarray = np.load(test_emb_path)
    print('Проверочные embeddings:', dset_emb.shape)
    print('Проверяемые embeddings:', test_emb.shape)

    # Если проверяется на датасете, у которого 1 семпл - 1 класс
    if len(test_emb.shape) == 2:
        test_emb = test_emb[:, None, :]

    # Нормализуем
    dset_emb = normalize(dset_emb)
    test_emb = normalize(test_emb)

    # Вычисляем центроиды классов датасета, с которыми будем сравниваться
    dset_centroids = dset_emb.mean(axis=1)
    
    n_cls, n_samples, embed_dim = test_emb.shape
    # Средние углы от семплов i-го класса до всех центроидов
    mean_angles = np.empty((n_cls, n_cls), dtype=np.float32)
    # Предсказания для всех семплов в виде индексов классов
    predicts = np.empty((n_cls, n_samples), np.int32)

    iterator = range(n_cls)
    if show_progress:
        desc = 'Вычисление угловых расстояний'
        iterator = tqdm(iterator, desc=desc)
    for cls_idx in iterator:
        # (n_samples, n_cls)
        cls_angles = angular_many2many(test_emb[cls_idx], dset_centroids)
        mean_angles[cls_idx] = cls_angles.mean(axis=0)  # (n_cls,)
        predicts[cls_idx] = cls_angles.argmin(axis=1)  # (n_samples)

    print('Angles:', mean_angles, sep='\n')
    print('Mean angle:', np.mean(mean_angles), sep='\n')

    ground_truth = np.arange(n_cls)
    ground_truth = np.tile(ground_truth, (n_samples, 1)).T
    cls_accuracy = np.stack([calculate_accuracy(predicts[i], ground_truth[i])
                             for i in range(n_cls)])

    print('Predicted classes', predicts, sep='\n')
    print("Classes' accuracy", cls_accuracy, sep='\n')
    print('Mean accuracy', np.mean(cls_accuracy), sep='\n')

    # Построение тепловой карты из классовой точности
    if map_size is not None and step is not None and win_size is not None:
        min_acc = 0.0
        max_acc = 1.0
        color_pallet = Colormap(min_acc, max_acc, colormap='Greys_r')
        heatmap = create_overlapping_heatmap(cls_accuracy, map_size, win_size,
                                             step, color_pallet, show_progress)
        plt.imshow(heatmap)
        plt.show()


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Протестировать модель. Посчитать метрики на основе '
                     'указанных embeddings.'))

    parser.add_argument(
        'dataset_embeddings', type=Path,
        help='Путь к файлу с embeddings от датасета.')
    parser.add_argument(
        'test_embeddings', type=Path,
        help='Путь к файлу с тестируемыми embeddings.')
    parser.add_argument(
        '--show_calc_progress', action='store_true',
        help='Отображать процесс вычислений.')
    
    heatmap_group = parser.add_argument_group(
        title='Настройки тепловой карты',
        description=('Параметры для создания тепловой карты '
                     'из получаемой классовой точности. '
                     'Если не передан хотя бы один из параметров, '
                     'то тепловая карта не будет строиться.'))
    heatmap_group.add_argument(
        '--map_h_w', type=int, nargs=2, default=None,
        help='Высота и ширина карты.')
    heatmap_group.add_argument(
        '--step', type=int, default=None,
        help='Размер шага окна в пикселях.')
    heatmap_group.add_argument(
        '--win_size', type=int, default=None,
        help='Размер окна в пикселях.')

    args = parser.parse_args()

    for path in {args.dataset_embeddings, args.test_embeddings}:
        if not path.exists():
            raise FileNotFoundError(
                f'Указанный файл с embeddings "{path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
