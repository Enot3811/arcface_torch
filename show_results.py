from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

from inference import inference


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def create_embeddings_plots(embeddings: np.ndarray):
    n_columns = 15
    n_rows = np.ceil(
        embeddings.shape[0] / n_columns).astype(np.int16)

    fig, axs = plt.subplots(n_rows, n_columns)
    fig.set_size_inches(20.0, 20.0, forward=True)
    fig.suptitle('Centroids embeddings.')
    for i in range(embeddings.shape[0]):
        row = i // n_columns
        column = i % n_columns

        axs[row][column].plot(embeddings[i])
    return fig


def calc_l2(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.sqrt((arr1 - arr2) ** 2)


def mean_angular_distances(
    embeddings: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """
    Берётся центроид и ембединги семплов каждого класса и вычисляется
    средний угол между ембедингами и центроидом соответствующего класса.

    Args:
        embeddings (np.ndarray): shape is `(n_classes, n_samples, embed_dim)`.
        centroids (np.ndarray): shape is `(n_classes, embed_dim)`.

    Returns:
        np.ndarray: shape is `(n_classes,)`
    """
    n_classes = embeddings.shape[0]
    classes_distances = []
    for cls in range(n_classes):
        cls_centroid = centroids[cls]
        cls_embeddings = embeddings[cls]
        classes_distances.append(
            angular_one2many(cls_centroid, cls_embeddings))

    classes_distances = np.array(classes_distances)
    mean_classes_distances = np.mean(classes_distances, axis=1)
    return mean_classes_distances


def angular_sample_to_centroids(embeddings: np.ndarray, centroids: np.ndarray):
    # Берётся случайный семпл из каждого класса и вычисляются расстояния
    # до всех центроидов
    # shape (n_classes, n_classes)
    n_classes, n_samples = embeddings.shape[:2]

    classes_distances = []
    for i in range(n_classes):
        j = random.randint(0, n_samples - 1)
        rand_sample = embeddings[i, j]

        classes_distances.append(
            angular_one2many(rand_sample, centroids))
    classes_distances = np.array(classes_distances)

    nearest_idxs = np.argmin(classes_distances, axis=1)
    print('Nearest indexes:', nearest_idxs, sep='\n')
    mask = np.arange(n_classes)
    correct = np.sum(nearest_idxs == mask)
    print('Correct:', correct, 'Total:', n_classes,
          'Accuracy', correct / n_classes)
    

def classify_samples(
    embeddings: np.ndarray, centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Return predicted indexes for every sample (shape n_cls x n_samples)
    # and accuracy per class (shape n_cls,)
    # embeddings shape (n_classes, )
    n_classes, n_samples = embeddings.shape[:2]

    classes_distances = []
    for i in range(n_classes):
        cur_distances = []
        for j in range(n_samples):
        
            current_sample = embeddings[i, j]

            cur_distances.append(angular_one2many(current_sample, centroids))
        classes_distances.append(np.array(cur_distances))
    classes_distances = np.array(classes_distances)

    nearest_idxs = np.argmin(classes_distances, axis=-1)
    gt_idxs = np.tile(np.arange(n_classes)[:, None], (1, 20))
    accuracy_per_class = np.sum(nearest_idxs == gt_idxs, axis=-1) / n_samples
    return nearest_idxs, accuracy_per_class
    

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
    cosine = np.dot(v1, v2.T)/(np.linalg.norm(v1) * np.linalg.norm(v2, axis=1))
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
    return np.arccos(cosine)


def main():
    # Get results embeddings
    embeddings_path = Path(
        __file__).parents[1] / 'data' / 'img_dataset' / 'results.npy'
    embeddings: np.ndarray = np.load(embeddings_path)

    norm_embed = normalize(embeddings)
    embed_centroids = norm_embed.mean(axis=1)

    n_classes, n_samples, embed_dim = embeddings.shape
    create_embeddings_plots(embed_centroids)

    
    mean_angles = mean_angular_distances(norm_embed, embed_centroids)

    print('Angles:', mean_angles, sep='\n')

    fig, axs = plt.subplots(1, 1)
    axs.plot(mean_angles)
    # plt.show()

    angular_sample_to_centroids(norm_embed, embed_centroids)


    # реальная картинка
    checkpoint_path = 'checkpoints/backbone.pth'
    model_name = 'r50'
    images = ['../data/test_sattelite_112x112.png']

    n_classes = embeddings.shape[0]
    results = [inference(checkpoint_path, model_name, img) for img in images]
    result = normalize(results[0]).squeeze()  # (1, 512)
    angles = angular_one2many(result, embed_centroids)
    
    print('Predicted class:', np.argmin(angles))


if __name__ == '__main__':
    main()
