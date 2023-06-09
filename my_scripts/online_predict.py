"""Скрипт для классификации участков, получаемых в реальном времени.

Скрипту указывается директория, в которую в реальном времени попадают снимки
с камеры. Последний сохранённый в директории снимок считывается и отправляется
в загруженную сеть.
"""


from pathlib import Path
import argparse
import time

import numpy as np
import torch
import cv2

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import normalize, classify_samples, normalize
from my_utils.model_tools import load_model, preprocess_model_input
from my_utils.image_tools import read_image, resize_image
from my_utils.grid_tools import ClassesGrid


def main(**kwargs):
    input_dir: Path = kwargs['input_dir']
    model_path: Path = kwargs['model_path']
    model_name: str = kwargs['model_name']
    embeddings_pth: Path = kwargs['embeddings']
    file_ext: str = kwargs['files_extension']
    show_time: bool = kwargs['time_info']
    cls_img_dir: Path = kwargs['cls_images']
    source_map_pth: Path = kwargs['map_path']
    step: int = kwargs['step_px']
    
    # (n_cls, n_samples, embed), или (n_cls, 1, embed), или (n_cls, embed)
    embeddings: np.ndarray = np.load(embeddings_pth)
    embeddings = normalize(embeddings)
    if len(embeddings.shape) == 3:
        embeddings = embeddings.mean(axis=1)
    
    if source_map_pth is not None:
        source_map = read_image(source_map_pth)
        cls_grid = ClassesGrid(*source_map.shape[:2], step)

    with torch.no_grad():
        model = load_model(model_name, model_path)
        img_paths = set(input_dir.glob('*.' + file_ext))
        while True:
            # Читаем все пути
            updated_paths = set(input_dir.glob('*.' + file_ext))
            # Отсеиваем старые для быстродействия
            new_paths = updated_paths - img_paths
            img_paths = updated_paths

            new_paths = list(new_paths)
            
            if len(new_paths) != 0:
                # Из оставшихся новых берём 1 самый последний
                new_paths.sort()
                pth = new_paths[-1]

                if show_time:
                    start = time.time()

                if file_ext == 'npy':
                    img = np.load(pth)
                else:
                    #TODO иногда выскакивает ошибка чтения
                    # Перезапуск помогает, но стоит разобраться
                    img = read_image(pth)
                model_input = preprocess_model_input(img)
                output: torch.Tensor = model(model_input)  # (1, embed_dim)
                img_embeddings = normalize(output.numpy())

                predicted_cls = classify_samples(img_embeddings, embeddings)[0]
                print('Predicted class:', predicted_cls)
                if show_time:
                    end = time.time() - start
                    print('{:.2f} секунд'.format(end))
                if source_map_pth is not None:
                    map_selected = cls_grid.show_selected_reg(
                        source_map, predicted_cls)
                    # Изменить размеры, чтобы отобразить в окне
                    show_img_w = 1000  # Ширина изображения в окне
                    h, w = map_selected.shape[:2]
                    resize_factor = show_img_w / w
                    show_img_h = int(h * resize_factor)
                    map_selected = resize_image(
                        map_selected, (show_img_h, show_img_w))

                    cv2.imshow('Predicted region',
                               cv2.cvtColor(map_selected, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(10) == 27:  # Клавиша Esc
                        break
                if cls_img_dir is not None:
                    cls_img = cv2.imread(
                        str(cls_img_dir / f's{predicted_cls}.jpg'))
                    cv2.imshow('Predicted class', cls_img)
                    if cv2.waitKey(10) == 27:  # Клавиша Esc
                        break


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Классификация изображений в реальном времени.'))

    parser.add_argument(
        'input_dir', type=Path,
        help='Путь к директории, из которой берутся снимки для классификации.')
    parser.add_argument(
        'model_path', type=Path,
        help='Путь к весам обученной модели.')
    parser.add_argument(
        'model_name', type=str,
        help='Название модели.')
    parser.add_argument(
        'embeddings', type=Path,
        help='Путь к файлу с embeddings для сравнения.')
    parser.add_argument(
        '--files_extension', type=str, default='jpg',
        choices=['jpg', 'png', 'npy'], help='Расширения читаемых файлов.')
    parser.add_argument(
        '--time_info', action='store_true',
        help='Отображать время, затрачиваемое на обработку изображения.')
    parser.add_argument(
        '--cls_images', type=Path, default=None,
        help=('Директория с изображениями предсказываемых классов. '
              'Если указана, то будет отображаться изображение '
              'предсказанного класса.'))
    parser.add_argument(
        '--map_path', type=Path, default=None,
        help=('Путь до изображения карты. Если указан, то классифицированный '
              'регион будет отображаться на карте.'))
    parser.add_argument(
        '--step_px', type=int, default=224,
        help=('Шаг окна в пикселях. '
              'По умолчанию равен 224, то есть без перекрытия.'))
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f'Указанная директория со снимками "{args.input_dir}" '
            'не существует.')
    if not args.embeddings.exists():
        raise FileNotFoundError(
            f'Указанный файл с embeddings "{args.embeddings}" не существует.')
    if not args.model_path.exists():
        raise FileNotFoundError(
            f'Файл с весами модели "{args.model_path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
