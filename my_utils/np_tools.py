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
    angles = np.empty((n, k))
    for i in range(n):
        angles[i] = angular_one2many(v1[i], v2, metric)
    return angles


def classify_samples(
    embeddings: np.ndarray, centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Классифицировать полученные embeddings вектора по полученным центроидам.
    
    Определение класса происходит посредством вычисления угловых расстояний
    между векторами из `embeddings` и векторами центроидов из `centroids`.
    Ближайший центроид и представляет предсказанный класс.

    Parameters
    ----------
    embeddings : np.ndarray
        Пакет embeddings векторов с размерами `[n_samples, embed_dim]`.
    centroids : np.ndarray
        Центроиды классов с размерами `[n_cls, embed_dim]`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Вектор с индексами предсказанных классов размером `[n_samples,]`.
    """
    distances = angular_many2many(embeddings, centroids)
    indexes = np.argmin(distances, axis=1)
    return indexes


def calculate_accuracy(
    predicts: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """Вычислить точность `predicts` по отношению к `ground_truth`.

    Parameters
    ----------
    predicts : np.ndarray
        Вектор предсказанных классов с размерностью `(n_samples,)`.
    ground_truth : np.ndarray
        Вектор истинных классов с размерностью `(n_samples,)`.

    Returns
    -------
    float
        Значение точности от 0.0 до 1.0.
    """    
    n_samples = predicts.shape[0]
    return np.sum(predicts == ground_truth) / n_samples
