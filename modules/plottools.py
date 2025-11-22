import numpy as np
import matplotlib.pyplot as plt


def add_wavelength_scale(image: np.array, scale_factor: int = 2):
    """
    Добавляет шкалу длин волн под изображением с настраиваемыми рисками
    """
    original_height, original_width = image.shape[:2]

    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    LAMBDA_MIN = 266.798  # длина волны при x=0
    LAMBDA_MAX = 0.745678 * (original_width - 1) + LAMBDA_MIN  # длина волны при x=639

    # создаем фигуру с отступом для шкалы
    fig = plt.figure(figsize=(new_width / 100, (new_height + 120) / 100))
    ax = fig.add_subplot(111)

    # отображаем изображение
    ax.imshow(image / np.max(image), extent=[0, new_width, 0, new_height])
    ax.axis("off")  # скрываем оси

    start_wl = ((LAMBDA_MIN // 50) + 1) * 50  # ближайшая отметка сверху
    end_wl = ((LAMBDA_MAX // 50) + 1) * 50
    wavelengths = np.arange(start_wl, end_wl + 1, 50)

    # Переводим длины волн в позиции пикселей
    x_positions = [
        (wl - LAMBDA_MIN) / 0.745678 for wl in wavelengths
    ]  # оригинальные координаты

    # Масштабируем координаты под увеличенное изображение
    scaled_x_positions = [x * scale_factor for x in x_positions]

    # Создаем шкалу
    scale_ax = fig.add_axes([0.1, 0.01, 0.8, 0.08])  # [left, bottom, width, height]
    scale_ax.set_xlim(0, new_width)
    scale_ax.axis("off")

    # Основная линия шкалы
    scale_ax.plot([0, new_width], [0.5, 0.5], color="black", linewidth=1)

    # Добавляем риски и метки
    for x, wl in zip(scaled_x_positions, wavelengths):
        if 0 <= x <= new_width:  # Проверка, чтобы риск не выходила за пределы
            # Вертикальная линия (от основной линии шкалы вверх)
            scale_ax.plot([x, x], [0.5, 0.6], color="black", linewidth=1)
            # Текст под риской
            scale_ax.text(x, 0.4, f"{int(wl)}", ha="center", va="top", fontsize=12)

    return fig
