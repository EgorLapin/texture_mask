"""
Скрипт для создания маски на основе текстуры с использованием модуля GLCM_analysis.

Предполагается, что GLCM_analysis.py находится в той же директории.
Создает составное изображение с текстурой кирпича и однородным фоном,
вычисляет локальный контраст с помощью скользящего окна и функций модуля,
применяет пороговую обработку для создания бинарной маски, применяет маску и отображает результаты.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte

import GLCM_analysis as ga


def compute_local_contrast(gray_img, window_size=3, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=256):
    """
    Вычисление локального контраста GLCM с использованием скользящего окна и функций модуля.

    Аргументы:
        gray_img: Изображение в градациях серого (uint8).
        window_size: Размер квадратного окна (нечётное число).
        distances: Расстояния для GLCM.
        angles: Углы для GLCM.
        levels: Количество уровней серого.

    Возвращает:
        ndarray: Карта контраста (того же размера, что и gray_img, границы нулевые).
    """
    try:
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size должен быть нечетным и >= 3")
        
        h, w = gray_img.shape
        r = window_size // 2
        contrast_map = np.zeros((h, w), dtype=float)
        for i in range(r, h - r):
            print(f"Обработка строки {i}/{h - r}...") 
            for j in range(r, w - r):
                patch = gray_img[i - r: i + r + 1, j - r: j + r + 1]
                glcm = ga.compute_glcm(patch, distances=distances, angles=angles, levels=levels)
                if glcm is not None:
                    contrast = ga.metric_contrast(glcm)
                    if not np.isnan(contrast):
                        contrast_map[i, j] = contrast
        return contrast_map
    except Exception as exc:
        print(f"[compute_local_contrast] Ошибка: {exc}")
        return None

def display_four_images(img1, img2, img3, img4, title1="Составное изображение", title2="Карта контраста", title3="Бинарная маска", title4="Маскированное изображение"):
    """
    Отображение четырех изображений в ряд с использованием matplotlib.

    Аргументы:
        img1: Составное изображение.
        img2: Карта контраста.
        img3: Бинарная маска.
        img4: Маскированное изображение.
        title1, title2, title3, title4: Заголовки.
    """
    try:
        plt.figure(figsize=(20, 5))

        # Составное изображение
        plt.subplot(1, 4, 1)
        gray1 = ga.prepare_gray(img1)
        plt.imshow(gray1, cmap="gray")
        plt.title(title1)
        plt.axis("off")

        # Карта контраста
        plt.subplot(1, 4, 2)
        plt.imshow(img2, cmap="jet")
        plt.title(title2)
        plt.axis("off")

        # Бинарная маска
        plt.subplot(1, 4, 3)
        plt.imshow(img3, cmap="gray")
        plt.title(title3)
        plt.axis("off")

        # Маскированное изображение
        plt.subplot(1, 4, 4)
        plt.imshow(img4, cmap="gray")
        plt.title(title4)
        plt.axis("off")

        plt.show()
    except Exception as exc:
        print(f"[display_four_images] Ошибка: {exc}")


def execute():
    """
    Основное выполнение: создание составного изображения, вычисление признаков, локального контраста, маски, отображение.
    """
    try:
        # Загрузка изображения кирпича с использованием модуля
        img_brick = ga.load_image("brick")
        if img_brick is None:
            raise RuntimeError("Не удалось загрузить изображение кирпича.")

        # Создание однородного фона
        img_uniform = np.full_like(img_brick, 128, dtype=np.uint8)

        # Создание составного изображения: левая половина — кирпич, правая — однородный фон
        h, w = img_brick.shape
        mid = w // 2
        composite = np.zeros((h, w), dtype=np.uint8)
        composite[:, :mid] = img_brick[:, :mid]
        composite[:, mid:] = img_uniform[:, mid:]

        # Вычисление глобальных признаков с использованием модуля для установки порога
        res_brick = ga.compute_glcm_features(composite[:, :mid], "brick_part")
        res_uniform = ga.compute_glcm_features(composite[:, mid:], "uniform_part")

        contrast_brick = res_brick[1].get("contrast", float('nan'))
        contrast_uniform = res_uniform[1].get("contrast", float('nan'))

        if np.isnan(contrast_brick) or np.isnan(contrast_uniform):
            raise RuntimeError("Не удалось вычислить глобальные контрасты.")

        threshold = (contrast_brick + contrast_uniform) / 2
        print(f"Порог установлен на: {threshold:.4f}")

        # Подготовка составного изображения в градациях серого
        gray_composite = ga.prepare_gray(composite)
        if gray_composite is None:
            raise RuntimeError("Не удалось подготовить составное изображение в градациях серого.")

        # Вычисление карты локального контраста
        contrast_map = compute_local_contrast(gray_composite)
        if contrast_map is None:
            raise RuntimeError("Не удалось вычислить карту локального контраста.")

        # Создание бинарной маски: 1, где контраст > порога
        binary_mask = (contrast_map > threshold).astype(np.uint8)

        # Применение маски к изображению
        masked_image = gray_composite * binary_mask

        # Отображение результатов
        display_four_images(composite, contrast_map, binary_mask, masked_image)

    except Exception as exc:
        print(f"[execute] Фатальная ошибка: {exc}")

if __name__ == "__main__":
    execute()