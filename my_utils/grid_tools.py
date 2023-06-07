"""Модуль с инструментами для работы с сеткой классов."""


from typing import List, Tuple, Union
from pathlib import Path
import sys

import numpy as np
import cv2
import seaborn as sns

sys.path.append(str(Path(__file__).parents[1]))
from my_utils.image_tools import resize_image


Rectangle = Tuple[Tuple[int, int], Tuple[int, int]]
Color = Tuple[int, int, int]


class ClassesGrid:
    def __init__(self, h_map: int, w_map: int, step: int, win: int = 224):
        """
        Инициализация сетки на основе размеров карты, окна и перекрывающего
        шага.

        Parameters
        ----------
        h_map : int
            Высота карты.
        w_map : int
            Ширина карты.
        step : int
            Шаг окна.
        win : int, optional
            Размер окна. По умолчанию равен 224.
        """
        # Каждый ij элемент сетки - это верхняя левая и нижняя правая точки,
        # представляющие собой прямоугольник, покрывающий ij класс
        self.cls_grid: List[List[Rectangle]] = []
        self.h_map = h_map
        self.w_map = w_map
        
        # Проходим с заданными шагами по размеру изображения
        cur_y = 0
        for i, cur_y in enumerate(range(0, h_map - win, step)):
            cur_x = 0
            self.cls_grid.append([])
            for cur_x in range(0, w_map - win, step):
                self.cls_grid[i].append((
                    (cur_y, cur_x),
                    (cur_y + win, cur_x + win)))
        self.n_h_win = len(self.cls_grid)
        self.n_w_win = len(self.cls_grid[0])

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Rectangle:
        """Взять классовый прямоугольник по индексу.

        Parameters
        ----------
        idx : Union[int, Tuple[int, int]]
            Индекс запрашиваемого класса. Может быть представлен как пара
            целых чисел, представляющие индексы окна вдоль y и x, или одно
            целое число, представляющее индекс окна при вытянутой карте вдоль
            одной оси.

        Returns
        -------
        Rectangle
            Пара точек, представляющих прямоугольник запрашиваемого класса.

        Raises
        ------
        TypeError
            Неверный тип индекса.
        """
        if isinstance(idx, tuple):
            i, j = idx
        elif isinstance(idx, (int, np.int16, np.int32, np.int64)):
            i = idx // self.n_w_win
            j = idx % self.n_w_win
        else:
            raise TypeError('Неверный тип индекса.')
        return self.cls_grid[i][j]
    
    def show_windows_grid(
        self,
        map_image: np.ndarray,
        color: Color = (255, 0, 0),
        thickness: int = 1
    ) -> np.ndarray:
        """Отобразить сетку на переданном изображении.

        Parameters
        ----------
        image : np.ndarray
            Изображение карты.
        win_size : int
            Размер окна.
        stride : int
            Шаг окна.
        color : Tuple[int, int, int], optional
            Цвет рисуемых рамок. По умолчанию - красный.
        thickness : int, optional
            Толщина рисуемых рамок. По умолчанию - 1.

        Returns
        -------
        np.ndarray
            Карта с отображённой сеткой.
        """
        # Копировать изображение, чтобы не испортить исходник
        map_image = map_image.copy()
        if map_image.shape[:2] != (self.h_map, self.w_map):
            map_image = resize_image(map_image, (self.h_map, self.w_map))

        for row in self.cls_grid:
            for rect in row:
                p1, p2 = rect
                cv2.rectangle(map_image, p1[::-1], p2[::-1], color, thickness)
        return map_image

    def show_selected_reg(
        self,
        map_image: np.ndarray,
        idx: Union[int, Tuple[int, int]],
        color: Color = (0, 200, 0),
        alpha: float = 0.4
    ) -> np.ndarray:
        """Показать выбранный регион на переданной карте.

        Parameters
        ----------
        map_image : np.ndarray
            Изображение карты.
        idx : Union[int, Tuple[int, int]]
            Индекс выделяемого класса.
        color : Color, optional
            Цвет для выделения, по умолчанию светло зелёный (0, 200, 0).
        alpha : float, optional
            Степень прозрачности выделения, по умолчанию 0.4.

        Returns
        -------
        np.ndarray
            Карта с отображённым выделенным регионом.
        """
        # Копировать изображение, чтобы не испортить исходник
        map_image = map_image.copy()
        if map_image.shape[:2] != (self.h_map, self.w_map):
            map_image = resize_image(map_image, (self.h_map, self.w_map))
        
        p1, p2 = self[idx]
        overlay = map_image.copy()
        cv2.rectangle(overlay, p1[::-1], p2[::-1], color, -1)

        # Накладываем изображение с прямоугольником на изображение без него
        map_image = cv2.addWeighted(overlay, alpha, map_image, 1 - alpha, 0)
        return map_image
    

class Colormap:
    """Colormap для заданного диапазона дробных чисел с указанной точностью.

    Позволяет брать цвет в RGB формате с распределением от 0.0 до 1.0 для
    чисел из указанного диапазона с указанным количеством знаков после запятой.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        accuracy: int = 2,
        colormap: str = 'Greys'
    ) -> None:
        """Инициализация colormap.

        Parameters
        ----------
        min_value : float
            Минимальное значение colormap.
        max_value : float
            Максимальное значение colormap.
        accuracy : int, optional
            Количество знаков после запятой у числе в colormap.
            По умолчанию равно двум.
        colormap : str, optional
           Название цветовой палитры из seaborn. По умолчанию "Greys".
        """
        self.accuracy = accuracy
        self.mult = 10 ** accuracy  # Множитель для перехода из float в int
        color_numbers = np.arange(
            min_value * self.mult, max_value * self.mult + 1, dtype=np.int32)
        num_colors = color_numbers.shape[0]
        self.colormap = sns.color_palette(colormap, num_colors)
        # Словарь с числом float домноженным на mult и конвертированным в int,
        # под которым хранится соответствующий цвет
        self.color_dict = {num: color
                           for num, color in zip(color_numbers, self.colormap)}
        
    def __getitem__(self, idx: float) -> Tuple[float, float, float]:
        """Взять цвет для числа.

        Parameters
        ----------
        idx : float
            Число для которого необходимо взять цвет.

        Returns
        -------
        Tuple[float, float, float]
            RGB цвет с распределением от 0.0 до 1.0.
        """
        return self.color_dict[int(np.round(idx, self.accuracy) * self.mult)]
