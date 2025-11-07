import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from reader import load_raw, read_cih, find_raw_dirs
from noise import subtract_dark

# Глобальные настройки
INPUT_DIR = '/home/ropptar/Downloads/charges/'
OUTPUT_DIR = Path(__file__).parent / "output"

print(OUTPUT_DIR)
SHEAR_MAGNITUDE = [32, 32, 32, 25]
CROP_CONFIG = [{
    'y_start_ratio': 0.28,
    'y_end_ratio': 0.5
}, {
    'y_start_ratio': 0.3,
    'y_end_ratio': 0.5
}, {
    'y_start_ratio': 0.67,
    'y_end_ratio': 0.8
}, {
    'y_start_ratio': 0.75,
    'y_end_ratio': 1
}]
DARK_CONFIG = [[0, 1, 7, 8, 9],
               [0, 8, 9],
               [0, 1, 2, 3, 4, 5, 11, 12, 13, 14],
               [0, 1, 2, 3, 13, 14, 15, 16, 17, 18, 19]]


def shear_image_y(image, shear_magnitude):
    """Вертикальный сдвиг с обнулением краев"""
    h, w = image.shape[:2]
    M = np.array([[1, 0, 0], [shear_magnitude / w, 1, 0]], dtype=np.float32)

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=0)


def add_wavelength_scale(image, scale_factor=2):
    """
    Добавляет шкалу длин волн под изображением с настраиваемыми рисками

    Args:
        image: numpy array изображения (height, width, channels)
        scale_factor: во сколько раз увеличивать изображение

    Returns:
        fig: matplotlib figure с изображением и шкалой
    """
    # Исходные параметры
    original_height, original_width = image.shape[:2]

    # Рассчитываем новые размеры изображения
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Параметры шкалы
    LAMBDA_MIN = 266.798  # длина волны при x=0
    LAMBDA_MAX = 0.745678 * (original_width - 1) + LAMBDA_MIN  # длина волны при x=639

    # Создаем фигуру с увеличенным размером
    fig = plt.figure(figsize=(new_width / 100, (new_height + 120) / 100))  # Увеличили нижний отступ для шкалы
    ax = fig.add_subplot(111)

    # Отображаем увеличенное изображение
    ax.imshow(image / np.max(image), extent=[0, new_width, 0, new_height])
    ax.axis('off')  # скрываем оси

    # Генерируем метки через каждые 50 нм
    start_wl = ((LAMBDA_MIN // 50) + 1) * 50  # ближайшая отметка сверху
    end_wl = ((LAMBDA_MAX // 50) + 1) * 50
    wavelengths = np.arange(start_wl, end_wl + 1, 50)

    # Переводим длины волн в позиции пикселей
    x_positions = [(wl - LAMBDA_MIN) / 0.745678 for wl in wavelengths]  # оригинальные координаты

    # Масштабируем координаты под увеличенное изображение
    scaled_x_positions = [x * scale_factor for x in x_positions]

    # Создаем шкалу
    scale_ax = fig.add_axes([0.1, 0.01, 0.8, 0.08])  # [left, bottom, width, height]
    scale_ax.set_xlim(0, new_width)
    scale_ax.axis('off')

    # Основная линия шкалы
    scale_ax.plot([0, new_width], [0.5, 0.5], color='black', linewidth=1)

    # Добавляем риски и метки
    for x, wl in zip(scaled_x_positions, wavelengths):
        if 0 <= x <= new_width:  # Проверка, чтобы риск не выходила за пределы
            # Вертикальная линия (от основной линии шкалы вверх)
            scale_ax.plot([x, x], [0.5, 0.6], color='black', linewidth=1)
            # Текст под риской
            scale_ax.text(x, 0.4, f'{int(wl)}', ha='center', va='top', fontsize=12)

    return fig
def process_raw_folder(raw_folder, count):
    """Обработка одной RAW-папки"""
    # Создаем пути для результатов
    print(raw_folder)
    output_folder = OUTPUT_DIR / f'{count + 1}'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Загрузка параметров изображения
    try:
        height, width, exposure = read_cih(raw_folder)
    except Exception as e:
        print(f"Error reading CIH in {raw_folder}: {str(e)}")
        return

    # Обработка RAW-файлов
    raw_paths = np.array(sorted(raw_folder.glob("*.raw")))
    raw_files = np.array([load_raw(raw_path, height, width) for raw_path in raw_paths])
    indices = np.array(DARK_CONFIG[count])

    mask = np.ones(len(raw_files), dtype=bool)
    mask[indices] = False
    dark_img = np.mean(raw_files[indices], axis=0)
    # print(dark_img.shape)
    # maxv = {'R': 0, 'G': 0, 'B': 0}
    # minv = {'R': 0, 'G': 0, 'B': 0}
    # mean = {'R': 0, 'G': 0, 'B': 0}
    # std_dev = {'R': 0, 'G': 0, 'B': 0}
    # color = {0:'R', 1:'G', 2:'B'}
    # for i in range(3):
    #     ch = dark_img[:, :, i]
    #     maxv[color[i]] = (np.max(ch))
    #     minv[color[i]] = (np.min(ch))
    #     mean[color[i]] = (np.mean(ch))
    #     std_dev[color[i]] = (np.std(ch))
    # print(maxv, minv, mean, std_dev)
    # for i in 'RGB':
    #     print(f'{i}: {mean[i]+std_dev[i]}')

    charge_images = raw_files[mask]
    #  ВРЕМЕННО ЗАМЕНИМ ВЫЧИТАНИЕ НА ВЫЧИТАНИЕ С ОБРЕЗКОЙ!!________________________________

    for idx, img in enumerate(charge_images):
        subtr_img = img - dark_img
        subtr_img = np.array(subtr_img)
        subtr_img = np.clip(subtr_img, 0, None)
        corrected = shear_image_y(subtr_img, SHEAR_MAGNITUDE[count])

        # Обрезка
        y_start = int(height * CROP_CONFIG[count]['y_start_ratio'])
        y_end = int(height * CROP_CONFIG[count]['y_end_ratio'])
        cropped = corrected[y_start:y_end, :]
        cropped = np.clip(cropped, 0, None)
        fig, axes = plt.subplots(2, 1, figsize=(8, 3))
        axes[0].imshow(img / np.max(img))
        axes[1].imshow(cropped / np.max(cropped))
        plt.savefig(output_folder / f"frame_{idx:02d}.png")
        plt.close()
        # Сохранение

        fig = add_wavelength_scale(img, )
        fig.savefig(output_folder / f"origframe_{idx:02d}.png", bbox_inches='tight', dpi=200)
        plt.close(fig)
    ''' 
    for idx, raw_path in enumerate(raw_files):
        try:
            # Загрузка и коррекция
            raw_img = load_raw(raw_path, height, width)

            # corrected = shear_image_y(raw_img, SHEAR_MAGNITUDE[count])

            # Обрезка
            y_start = int(height * CROP_CONFIG[count]['y_start_ratio'])
            y_end = int(height * CROP_CONFIG[count]['y_end_ratio'])
            # cropped = corrected[y_start:y_end, :]

            # Сохранение
            # np.save(output_folder / f"frame_{idx:02d}.npy", cropped.astype(np.float32))
            corrected = cropped = None
            visualize_process(raw_img, corrected, cropped, exposure, count, output_folder / f"preview_{idx:02d}.png")
            # Визуализация для первых 5 кадров
            # if 3 < idx < 8:
            #

        except Exception as e:
            print(f"Error processing {raw_path}: {str(e)}")'''


def visualize_process(original, corrected, dark, exposure, count, save_path):
    """Генерация сравнительных графиков"""
    fig, axes = plt.subplots(8, 2, figsize=(24, 16))

    maxv = np.max(original)
    orig_norm = (original / maxv)
    h, w, c = orig_norm.shape
    r, g, b = cv2.split(orig_norm)
    axes[0, 0].imshow(orig_norm)
    axes[1, 0].imshow(r)
    axes[2, 0].imshow(g)
    axes[3, 0].imshow(b)
    for idx, ch in enumerate([r, g, b]):
        dft = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # print(dft_shift.shape)
        dft_img = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        axes[1 + idx, 1].imshow(dft_img)
    dark_norm = dark / np.max(dark)
    r, g, b = cv2.split(dark_norm)
    axes[4, 0].imshow(dark_norm)
    axes[5, 0].imshow(r)
    axes[6, 0].imshow(g)
    axes[7, 0].imshow(b)
    for idx, ch in enumerate([r, g, b]):
        dft = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # print(dft_shift.shape)
        dft_img = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        axes[5 + idx, 1].imshow(dft_img)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    '''
    maxv = np.max(original)
    s = np.mean(np.mean(original, axis=0), axis=0) / 65520
    sc = np.mean(np.mean(corrected, axis=0), axis=0) / 65520
    # Нормализация для визуализации
    fft_image = np.fft.fft2(original)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    magnitude_spectrum /= np.max(magnitude_spectrum)

    orig_norm = original / maxv
    corr_norm = corrected / np.max(corrected)
    dark_norm = dark / np.max(dark)
    # Оригинал
    try:
        axes[0].imshow(orig_norm)
        axes[0].set_title(
            f"Original Image, t={count * exposure:.2e}, R={s[0]:.2e}, G={s[1]:.2e}, B={s[2]:.2e},  s ={np.mean(s, axis=0):.2e}")
        axes[0].axis('off')

        # # Скорректированное
        axes[1].imshow(corr_norm)
        axes[1].set_title(f"Dark subtract, R={sc[0]:.2e}, G={sc[1]:.2e}, B={sc[2]:.2e},  s ={np.mean(sc, axis=0):.2e}")
        axes[1].axis('off')
        # axes[1].imshow(corr_norm)
        # axes[1].set_title(f"Corrected (Shear Y={SHEAR_MAGNITUDE[count]}px)")
        # axes[1].axis('off')

        # Темное
        axes[2].imshow(dark_norm)
        axes[2].set_title("Dark")
        axes[2].axis('off')
        # # Обрезанное
        # axes[2].imshow(crop_norm)
        # axes[2].set_title("Cropped Strip")
        # axes[2].axis('off')
        axes[3].imshow(magnitude_spectrum)
        axes[3].set_title("FFT orig")
        axes[3].axis('off')
        fft_image = np.fft.fft2(corrected)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        magnitude_spectrum /= np.max(magnitude_spectrum)
        axes[4].imshow(magnitude_spectrum)
        axes[4].set_title("FFT corr")
        axes[4].axis('off')

        fft_image = np.fft.fft2(dark)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        magnitude_spectrum /= np.max(magnitude_spectrum)
        axes[5].imshow(magnitude_spectrum)
        axes[5].set_title("FFT dark")
        axes[5].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        '''


def process_channel(channel):
    # Прямое преобразование Фурье
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Создание маски для удаления вертикальных полос (горизонтального шума)
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[:, ccol - 10:ccol + 10] = 0  # Вертикальная полоса в центре

    # Применение маски
    fshift = dft_shift * mask

    # Обратное преобразование
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


def first_step():
    # Поиск всех RAW-папок
    input_root = INPUT_DIR
    raw_folders = find_raw_dirs(input_root)

    if not raw_folders:
        print("No raw folders found!")
        return

    # Обработка всех папок
    count = 0
    for folder in raw_folders:
        print(f"\nProcessing folder: {folder}")
        if count == 3:
            process_raw_folder(Path(folder), count)
        count += 1

    print(f"\nAll done! Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    first_step()
