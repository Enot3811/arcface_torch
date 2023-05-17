"""
Функции работающие с np массивами.
"""

import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Привести массив к диапазону от 0 до 1.

    Args:
        arr (np.ndarray): Исходный массив.

    Returns:
        np.ndarray: Нормализованный массив
    """    
    return (arr - arr.min()) / (arr.max() - arr.min())


def angular_one2many(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate angles in radians between a vector v1 and a batch of vectors v2.

    Args:
        v1 (np.ndarray): The vector with shape `(vector_dim,)`.
        v2 (np.ndarray): The batch of vectors with shape
        `(n_vectors, vector_dim)`.

    Returns:
        np.ndarray: The angles array with shape `(n_vectors,)`.
    """
    cosine = np.dot(v1, v2.T) / (np.linalg.norm(v1) *
                                 np.linalg.norm(v2, axis=1))
    # Избавляемся от погрешности
    cosine = np.clip(cosine, 0.0, 1.0)
    return np.arccos(cosine)


def angular_many2many(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate one to one angles in radians between batches of vectors
    v1 and v2.
    First element of v1 with first element of v2, second with second e.t.c.

    Args:
        v1 (np.ndarray): The batch of vectors with shape
        `(n_vectors, vector_dim)`.
        v2 (np.ndarray): The batch of vectors with shape
        `(n_vectors, vector_dim)`.

    Returns:
        np.ndarray: The angles array with shape `(n_vectors,)`
    """
    cosine = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) *
                                        np.linalg.norm(v2, axis=1))
    # Избавляемся от погрешности
    cosine = np.clip(cosine, 0.0, 1.0)
    return np.arccos(cosine)