from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from image_tools import (
    read_image, resize_image, process_raw_real_image,
    get_sliding_windows, get_scaled_shape, show_grid)
from torch_tools import get_augmentation, tensor_to_numpy, numpy_to_tensor


def main():
    """
    Здесь создаётся датасет из предоставленных реальных снимков местности.
    Был выбран 75 снимок для нарезания.
    """
    show = False
    save = True
    img_path = (Path(__file__).parents[1] /
                'data' / 'real_data_raw' / 'B075.jpg')
    img = read_image(img_path)
    img = process_raw_real_image(img)  # Крутим и отрезаем белые края

    num_samples = 500  # Количество производимых семплов на класс
    h_orig, w_orig = img.shape[:2]
    orig_scale = 0.5  # Метров в пикселе
    net_input_size = 112
    fov = 1000  # Сторона квадрата поля зрения в метрах
    overlap = 1000  # Шаг перекрывающего окна в метрах
    # Директория для сохранения изображений датасета
    new_dataset_path = (
        Path(__file__).parents[1] / 'data' / 'real_images_dataset' /
        f'{fov}m_{overlap}m_{num_samples}img' / 'images')

    print('Исходное изображение:', f'Размеры: {h_orig}, {w_orig}',
          f'Масштаб: {orig_scale} м/px', f'Поле зрения: {fov} м',
          f'Перекрывающий шаг: {overlap} м', sep='\n')
    
    h, w, scaled_overlap_px, new_scale = get_scaled_shape(
        h_orig, w_orig, orig_scale, overlap, fov, net_input_size)
    new_fov = net_input_size
    
    print('\nОбработанное изображение:', f'Размеры: {h}, {w}',
          f'Масштаб: {new_scale} м/px', f'Поле зрения: {new_fov} px',
          f'Перекрывающий шаг: {scaled_overlap_px} px', sep='\n')

    img = resize_image(img, (w, h))

    windows = get_sliding_windows(
        img, new_fov, new_fov, scaled_overlap_px)
    
    print('Порезанные окна:', windows.shape)

    augmentations = get_augmentation(color_jitter=True, elastic=True)
    windows = numpy_to_tensor(windows)

    # Отобразить порезанные окна
    # Чтобы корректно работало, необходимо резать без перекрытия
    if show:
        plt.imshow(img)

        n_h_win = h // new_fov
        n_w_win = w // new_fov
        for _ in range(6):
            augmented_windows = augmentations(windows)
            augmented_windows = tensor_to_numpy(augmented_windows)
            show_grid(augmented_windows, n_h_win, n_w_win)
            plt.show()


    # Создаём директории под классы
    if save:
        for i in range(windows.shape[0]):
            dir_path = new_dataset_path / f's{i}'
            dir_path.mkdir(parents=True, exist_ok=True)

        # Делаем случайные аугментации и сохраняем их
        b_size = 1225
        for i in tqdm(range(num_samples)):
            for j in range(0, windows.shape[0], b_size):

                augmented_windows = augmentations(windows[j:j + b_size])
                augmented_windows = tensor_to_numpy(augmented_windows)

                for k in range(j, j + b_size):
                    path = new_dataset_path / f's{k}' / f'{i}.jpg'
                    
                    cv2.imwrite(str(path), cv2.cvtColor(
                        augmented_windows[k % b_size], cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()