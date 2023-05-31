"""Функции работающие с np массивами."""

from typing import Tuple

import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Привести массив к диапазону от 0 до 1.

    Parameters
    ----------
    arr : np.ndarray
        Исходный массив.

    Returns
    -------
    np.ndarray
        Нормализованный массив.
    """    
    return (arr - arr.min()) / (arr.max() - arr.min())


def angular_one2many(
    v1: np.ndarray,
    v2: np.ndarray,
    metric: str = 'deg'
) -> np.ndarray:
    """Вычислить угол в радианах между вектором `v1` и пакетом векторов `v2`.

    Parameters
    ----------
    v1 : np.ndarray
        Вектор с размерами `(vector_dim,)`.
    v2 : np.ndarray
        Пакет векторов с размерами `(n_vectors, vector_dim)`.
    metric : str, optional
        Метрика измерения угла. Принимает "rad" для радианов и "deg" для
        градусов. По умолчанию равняется "deg".

    Returns
    -------
    np.ndarray
        Углы между векторами с размерностью `(n_vectors,)`.

    Raises
    ------
    ValueError
        Значение metric должно быть либо "rad", либо "deg".
    """
    if metric == 'rad':
        transf_coef = 1.0
    elif metric == 'deg':
        transf_coef = 180.0 / np.pi
    else:
        raise ValueError('Значение metric должно быть либо "rad", либо "deg", '
                         f'однако получено {metric}')
    cosine = np.dot(v1, v2.T) / (np.linalg.norm(v1) *
                                 np.linalg.norm(v2, axis=1))
    # Избавляемся от погрешности
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine) * transf_coef


def angular_many2many(
    v1: np.ndarray,
    v2: np.ndarray,
    metric: str = 'deg'
) -> np.ndarray:
    """Вычислить углы в радианах между пакетами векторов `v1` и `v2`.

    Parameters
    ----------
    v1 : np.ndarray
        Первый пакет векторов с размерами `(n_vectors, vector_dim)`.
    v2 : np.ndarray
        Второй пакет векторов с размерами `(k_vectors, vector_dim)`.
    metric : str, optional
        Метрика измерения угла. Принимает "rad" для радианов и "deg" для
        градусов. По умолчанию равняется "deg".

    Returns
    -------
    np.ndarray
        Углы между векторами с размерами `(n_vectors, k_vectors)`, где каждая
        n-я строка - это углы между n-м вектором с соответствующими векторами
        из `v2`.

    Raises
    ------
    ValueError
        Значение metric должно быть либо "rad", либо "deg".
    """
    if metric == 'rad':
        transf_coef = 1.0
    elif metric == 'deg':
        transf_coef = 180.0 / np.pi
    else:
        raise ValueError('Значение metric должно быть либо "rad", либо "deg", '
                         f'однако получено {metric}')
    n = v1.shape[0]
    k = v2.shape[0]
    cosines = np.empty((n, k))
    for i in range(n):
        cosines[i] = angular_one2many(v1[i], v2)
    # Избавляемся от погрешности
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.arccos(cosines) * transf_coef


def mean_angular_distances(
    embeddings: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """
    Вычислить среднее угловое расстояние между семплами класса и их центроидом.

    Args:
        embeddings (np.ndarray): Матрица embeddings с размерами
        `(n_cls, n_cls_samples, embed_dim)`.
        centroids (np.ndarray): Матрица с embeddings центроидов размерами
        `(n_cls, embed_dim)`.

    Returns:
        np.ndarray: Среднее угловое расстояние для каждого класса размером
        `(n_cls,)`
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
        `[n_cls, n_cls_samples, embed_dim]`.
        centroids (np.ndarray): A centroids tensor with shape
        `[n_cls, embed_dim]`.

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
