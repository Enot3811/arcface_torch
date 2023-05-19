"""
Скрипт для тестирования модели.

Тестирование проводится на основе сохранённых заранее npy файлов с embeddings.
"""


import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import normalize, angular_one2many


def mean_angular_distances(
    embeddings: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """
    Берётся центроид и embeddings семплов каждого класса и вычисляется
    средний угол между embeddings и центроидом соответствующего класса.

    Args:
        embeddings (np.ndarray): Матрица embeddings с размерами
        `(n_classes, n_samples, embed_dim)`.
        centroids (np.ndarray): Матрица с embeddings центроидов размерами
        `(n_classes, embed_dim)`.

    Returns:
        np.ndarray: Среднее угловое расстояние для каждого класса размером
        `(n_classes,)`
    """    
    n_classes = embeddings.shape[0]
    classes_distances = []
    for cls in range(n_classes):
        cls_centroid = centroids[cls]
        cls_embeddings = embeddings[cls]
        classes_distances.append(
            angular_one2many(cls_centroid, cls_embeddings))

    classes_distances = np.stack(classes_distances, axis=0)
    mean_classes_distances = np.mean(classes_distances, axis=1)
    return mean_classes_distances


def classify_samples(
    embeddings: np.ndarray, centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return predicted indexes for an every sample and accuracy per class.

    Args:
        embeddings (np.ndarray): An embeddings tensor with shape
        `[n_classes, embed_dim]`.
        centroids (np.ndarray): A centroids tensor with shape
        `[n_classes, embed_dim]`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The predicted indexes with shape
        `[n_cls, n_samples]` and accuracy per class with shape `[n_cls,]`.
    """
    n_classes, n_samples = embeddings.shape[:2]

    classes_distances = []
    for i in range(n_classes):
        cur_distances = []
        for j in range(n_samples):
        
            current_sample = embeddings[i, j]
            cur_distances.append(angular_one2many(current_sample, centroids))

        classes_distances.append(np.stack(cur_distances, axis=0))
    classes_distances = np.stack(classes_distances, axis=0)

    nearest_idxs = np.argmin(classes_distances, axis=-1)
    gt_idxs = np.tile(np.arange(n_classes)[:, None], (1, n_samples))
    accuracy_per_class = np.sum(nearest_idxs == gt_idxs, axis=-1) / n_samples
    return nearest_idxs, accuracy_per_class


def main(**kwargs):
    dset_emb_path = kwargs['dataset_embeddings']
    test_emb_path = kwargs['test_embeddings']

    dset_emb: np.ndarray = np.load(dset_emb_path)  # (n_cls, n_samples, embed)
    test_emb: np.ndarray = np.load(test_emb_path)

    # Если проверяется на датасете, у которого 1 семпл - 1 класс
    if len(test_emb.shape) == 2:
        test_emb = test_emb[:, None, :]

    # Нормализуем
    dset_emb = normalize(dset_emb)
    test_emb = normalize(test_emb)

    # Вычисляем центроиды классов датасета, с которыми будем сравниваться
    dset_centroids = dset_emb.mean(axis=1)
    
    # Средние углы между семплами и их центроидами.
    mean_angles = mean_angular_distances(test_emb, dset_centroids)
    print('Angles:', mean_angles, sep='\n')
    print('Mean angle:', np.mean(mean_angles), sep='\n')

    # Делаем predict, получаем предсказанные классы и среднюю точность
    predicted_cls, cls_accuracy = classify_samples(test_emb, dset_centroids)
    print('Predicted classes', predicted_cls, sep='\n')
    print("Classes' accuracy", cls_accuracy, sep='\n')
    print('Mean accuracy', np.mean(cls_accuracy), sep='\n')


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
    args = parser.parse_args()

    for path in {args.dataset_embeddings, args.test_embeddings}:
        if not path.exists():
            raise FileNotFoundError(
                f'Указанный файл с embeddings "{path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
