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


def angular_one2many(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate angles in radians between a vector v1 and a batch of vectors v2.

    Parameters
    ----------
    v1 : np.ndarray
        The vector with shape `(vector_dim,)`.
    v2 : np.ndarray
        The batch of vectors with shape `(n_vectors, vector_dim)`.

    Returns
    -------
    np.ndarray
        The angles array with shape `(n_vectors,)`.
    """
    cosine = np.dot(v1, v2.T) / (np.linalg.norm(v1) *
                                 np.linalg.norm(v2, axis=1))
    # Избавляемся от погрешности
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)


def angular_many2many(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate one to one angles in radians between batches of vectors
    v1 and v2.
    First element of v1 with first element of v2, second with second e.t.c.

    Parameters
    ----------
    v1 : np.ndarray
        The batch of vectors with shape `(n_vectors, vector_dim)`.
    v2 : np.ndarray
        The batch of vectors with shape `(n_vectors, vector_dim)`.

    Returns
    -------
    np.ndarray
        The angles array with shape `(n_vectors,)`.
    """
    cosine = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) *
                                        np.linalg.norm(v2, axis=1))
    # Избавляемся от погрешности
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)


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
