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
from my_utils.image_tools import save_image


def main(**kwargs):
    dset_emb_path: Path = kwargs['dataset_embeddings']
    test_emb_path: Path = kwargs['test_embeddings']
    show_progress: bool = kwargs['show_calc_progress']
    map_size: Optional[Tuple[int, int]] = kwargs['map_h_w']
    step: Optional[int] = kwargs['step']
    win_size: Optional[int] = kwargs['win_size']
    show_heatmap: bool = kwargs['show_heatmap']

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

    files_prefix = (f'{dset_emb_path.name.split(".")[0]}_'
                    f'{test_emb_path.name.split(".")[0]}_')

    print('Angles:', mean_angles, sep='\n')
    mean_angle_total = np.mean(mean_angles)
    print('Mean angle:', mean_angle_total, sep='\n')
    angles_path = dset_emb_path.parents[1] / f'{files_prefix}angles.txt'
    np.savetxt(angles_path, mean_angles)
    with open(angles_path, 'a') as f:
        f.write(f'Mean angle: {mean_angle_total}')
    

    ground_truth = np.arange(n_cls)
    ground_truth = np.tile(ground_truth, (n_samples, 1)).T
    cls_accuracy = np.stack([calculate_accuracy(predicts[i], ground_truth[i])
                             for i in range(n_cls)])

    print('Predicted classes', predicts, sep='\n')
    print("Classes' accuracy", cls_accuracy, sep='\n')
    mean_acc = np.mean(cls_accuracy)
    print('Mean accuracy', mean_acc, sep='\n')
    predicts_path = dset_emb_path.parents[1] / f'{files_prefix}predicts.txt'
    accuracy_path = dset_emb_path.parents[1] / f'{files_prefix}accuracy.txt'
    np.savetxt(predicts_path, predicts)
    np.savetxt(accuracy_path, cls_accuracy)
    with open(accuracy_path, 'a') as f:
        f.write(f'Mean accuracy: {mean_acc}')

    # Построение тепловой карты из классовой точности
    if map_size is not None and step is not None and win_size is not None:
        min_acc = 0.0
        max_acc = 1.0
        color_pallet = Colormap(min_acc, max_acc, colormap='Greys_r')
        heatmap = create_overlapping_heatmap(cls_accuracy, map_size, win_size,
                                             step, color_pallet, show_progress)

        img_path = dset_emb_path.parents[1] / 'acc_heatmap.jpg'
        save_image((heatmap * 255).astype(np.uint8), img_path)
        if show_heatmap:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(heatmap)
            plt.show()


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Протестировать модель. Посчитать метрики на основе '
                     'указанных embeddings. Создать тепловую карту точности'))

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
    heatmap_group.add_argument(
        '--show_heatmap', action='store_true',
        help='отобразить изображение тепловой карты.')

    args = parser.parse_args()

    for path in {args.dataset_embeddings, args.test_embeddings}:
        if not path.exists():
            raise FileNotFoundError(
                f'Указанный файл с embeddings "{path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
