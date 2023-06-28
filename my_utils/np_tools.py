"""Функции работающие с np массивами."""

from typing import Tuple, Union

import numpy as np
from tqdm import tqdm
import torch

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
    v1: Union[np.ndarray, torch.Tensor],
    v2: Union[np.ndarray, torch.Tensor],
    metric: str = 'deg'
) -> Union[np.ndarray, torch.Tensor]:
    """Вычислить угол в радианах между вектором `v1` и пакетом векторов `v2`.

    Вычисления производятся на том устройстве,
    которое поддерживается переданными массивами.
    Например, если массивы являются `torch.Tensor` на cuda,
    то вычисления будут точно так же проводиться через torch на cuda,
    и результат возвращается в том же формате.
    Если переданы `np.ndarray`, то вычисления проводятся через numpy на cpu.

    Parameters
    ----------
    v1 : Union[np.ndarray, torch.Tensor]
        Вектор с размерами `(vector_dim,)`.
    v2 : Union[np.ndarray, torch.Tensor]
        Пакет векторов с размерами `(n_vectors, vector_dim)`.
    metric : str, optional
        Метрика измерения угла. Принимает "rad" для радианов и "deg" для
        градусов. По умолчанию равняется "deg".

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Углы между векторами с размерностью `(n_vectors,)`.

    Raises
    ------
    ValueError
        Значение metric должно быть либо "rad", либо "deg".
    TypeError
        v1 и v2 должны оба быть либо np.ndarray, либо torch.Tensor.
    """
    if metric == 'rad':
        transf_coef = 1.0
    elif metric == 'deg':
        transf_coef = 180.0 / np.pi
    else:
        raise ValueError('Значение metric должно быть либо "rad", либо "deg", '
                         f'однако получено {metric}')
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        cosine = v1.dot(v2.T) / (
            np.linalg.norm(v1) * np.linalg.norm(v2, axis=1))
        # Избавляемся от погрешности
        cosine = np.clip(cosine, -1.0, 1.0, out=cosine)
        angles = np.arccos(cosine, out=cosine)
        angles *= transf_coef
    elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        cosine = v1.matmul(v2.T) / (torch.norm(v1) * torch.norm(v2, dim=1))
        cosine.clip_(-1.0, 1.0)
        angles = torch.arccos(cosine, out=cosine)
        angles *= transf_coef
    else:
        raise TypeError(
            'v1 и v2 должны оба быть либо np.ndarray, либо torch.Tensor.'
            f'Однако они имеют тип {type(v1)} и {type(v2)}.')
    return angles


def angular_many2many(
    v1: Union[np.ndarray, torch.Tensor],
    v2: Union[np.ndarray, torch.Tensor],
    metric: str = 'deg',
    show_progress: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Вычислить углы в радианах между пакетами векторов `v1` и `v2`.

    Вычисления производятся на том устройстве,
    которое поддерживается переданными массивами.
    Например, если массивы являются `torch.Tensor` на cuda,
    то вычисления будут точно так же проводиться через torch на cuda,
    и результат возвращается в том же формате.
    Если переданы `np.ndarray`, то вычисления проводятся через numpy на cpu.

    Parameters
    ----------
    v1 : Union[np.ndarray, torch.Tensor]
        Первый пакет векторов с размерами `(n_vectors, vector_dim)`.
    v2 : Union[np.ndarray, torch.Tensor]
        Второй пакет векторов с размерами `(k_vectors, vector_dim)`.
    metric : str, optional
        Метрика измерения угла. Принимает "rad" для радианов и "deg" для
        градусов. По умолчанию равняется "deg".
    show_progress : bool, optional
        Отображать прогресс вычислений. По умолчанию - `False`.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Углы между векторами с размерами `(n_vectors, k_vectors)`, где каждая
        n-я строка - это углы между n-м вектором с соответствующими векторами
        из `v2`.

    Raises
    ------
    ValueError
        Значение metric должно быть либо "rad", либо "deg".
    TypeError
        v1 и v2 должны оба быть либо np.ndarray, либо torch.Tensor.
    """
    if metric not in {'rad', 'deg'}:
        raise ValueError('Значение metric должно быть либо "rad", либо "deg", '
                         f'однако получено {metric}')
    n = v1.shape[0]
    k = v2.shape[0]

    if show_progress:
        desc = 'Вычисление угловых расстояний'
        iterator = tqdm(range(n), desc=desc)
    else:
        iterator = range(n)

    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        angles = np.empty((n, k))
        for i in iterator:
            angles[i] = angular_one2many(v1[i], v2, metric)
    elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        device = v1.device
        angles = torch.empty((n, k), dtype=torch.float32, device=device)
        for i in iterator:
            angles[i] = angular_one2many(v1[i], v2, metric)
    else:
        raise TypeError(
            'v1 и v2 должны оба быть либо np.ndarray, либо torch.Tensor.'
            f'Однако они имеют тип {type(v1)} и {type(v2)}.')
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
    predicts: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor]
) -> float:
    """Вычислить точность `predicts` по отношению к `ground_truth`.

    Parameters
    ----------
    predicts : Union[np.ndarray, torch.Tensor]
        Вектор предсказанных классов с размерностью `(n_samples,)`.
    ground_truth : Union[np.ndarray, torch.Tensor]
        Вектор истинных классов с размерностью `(n_samples,)`.

    Returns
    -------
    float
        Значение точности от 0.0 до 1.0.
    """    
    n_samples = predicts.shape[0]
    return (predicts == ground_truth).sum() / n_samples
