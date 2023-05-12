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
    img_path = (Path(__file__).parents[1] /
                'data' / 'real_data_raw' / 'B075.jpg')
    img = read_image(img_path)
    img = process_raw_real_image(img)  # Крутим и отрезаем белые края


    h_orig, w_orig = img.shape[:2]
    orig_scale = 0.5  # Метров в пикселе
    net_input_size = 112
    fov = 200  # Сторона квадрата поля зрения в метрах
    overlap = 30  # Шаг перекрывающего окна в метрах
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
    # plt.imshow(img)
    # plt.show()

    windows = get_sliding_windows(
        img, new_fov, new_fov, scaled_overlap_px)
    
    print('Порезанные окна:', windows.shape)

    # # Отобразить порезанные окна
    # # Чтобы корректно работало, необходимо резать без перекрытия
    # n_h_win = h // new_fov
    # n_w_win = w // new_fov
    # augmentations = get_augmentation()
    # windows = numpy_to_tensor(windows)
    # for _ in range(6):
    #     augmented_windows = augmentations(windows)
    #     augmented_windows = tensor_to_numpy(augmented_windows)
    #     show_grid(augmented_windows, n_h_win, n_w_win)
    #     plt.show()


    # Создаём директории под классы
    for i in range(windows.shape[0]):
        dir_path = (Path(__file__).parents[1] / 'data' /
                    'real_images_dataset' / f's{i}')
        dir_path.mkdir(parents=True, exist_ok=True)

    # Делаем случайные аугментации и сохраняем их
    augmentations = get_augmentation()
    windows = numpy_to_tensor(windows)
    b_size = 256
    for i in tqdm(range(100)):
        for j in range(0, windows.shape[0], b_size):

            augmented_windows = augmentations(windows[j:j + b_size])
            augmented_windows = tensor_to_numpy(augmented_windows)

            for k in range(j, j + b_size):
                path = (Path(__file__).parents[1] / 'data' / 'real_images_dataset'
                        / f's{k}' / f'{i}.jpg')
                
                cv2.imwrite(str(path), cv2.cvtColor(augmented_windows[j % b_size],
                                                    cv2.COLOR_RGB2BGR))
            


# 1px = 0.5m
# 5 на 5 (200m x 200m)
# шаг в 30m


if __name__ == '__main__':
    main()