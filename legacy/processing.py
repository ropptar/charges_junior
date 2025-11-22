import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import json
from scipy.interpolate import interp1d
import pandas as pd

FPS_CONFIG = {
    "1": 50000,
    "2": 50000,
    "3": 50000,
    "4": 102400,
}

dR = 1112
dG = 919
dB = 1757

lines = [656.279, 486.135, 434.0472, 410.1734]
OUTPUT_DIR = Path(__file__).parent / "output"

CALIBRATIONS = [
    {  # Для папки 1
        "calibration_points": {
            "522": 656.0,
            "432": 589.0,
            "294": 486.0
        }
    },
    {  # Для папки 2
        "calibration_points": {
            "522": 656.0,
            "432": 589.0,
            "294": 486.0
        }
    },
    {  # Для папки 3
        "calibration_points": {
            "522": 656.0,
            "432": 589.0,
            "294": 486.0
        }
    },
    {  # Для папки 4
        "calibration_points": {
            "522": 656.0,
            "432": 589.0,
            "294": 486.0
        }
    }
]
TRANS_PATH = Path(__file__).parent / "transmission.csv"


def get_calibration(discharge_number: int):
    """Получение калибровочных данных для разряда"""
    if 1 <= discharge_number <= 4:
        return CALIBRATIONS[discharge_number - 1]
    raise ValueError(f"No calibration for discharge {discharge_number}")


def calculate_lambda_coeffs(calibration_data: dict) -> tuple:
    """Рассчет коэффициентов линейного преобразования"""
    x = np.array(list(map(int, calibration_data["calibration_points"].keys())))
    y = np.array(list(calibration_data["calibration_points"].values()))
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def scale():
    """Загрузка данных чувствительности с погрешностями"""
    ds = pd.read_csv("sensitivity.csv", sep=";")
    dt = pd.read_csv("transmission.csv", decimal=',')

    wvl1, wvl2 = ds['Wavelength'], dt['Wavelength']

    # Создаем интерполяторы для средних значений и границ погрешностей
    SR = interp1d(wvl1, ds['SR'], kind='linear', fill_value='extrapolate')
    SG = interp1d(wvl1, ds['SG'], kind='linear', fill_value='extrapolate')
    SB = interp1d(wvl1, ds['SB'], kind='linear', fill_value='extrapolate')

    # Добавляем интерполяторы для относительных погрешностей
    E_R = interp1d(wvl1, ds['E_R'], kind='linear', fill_value='extrapolate')
    E_G = interp1d(wvl1, ds['E_G'], kind='linear', fill_value='extrapolate')
    E_B = interp1d(wvl1, ds['E_B'], kind='linear', fill_value='extrapolate')

    T = interp1d(wvl2, dt['Total'], kind='linear', fill_value='extrapolate')

    # Возвращаем все интерполяторы
    return (SR, SG, SB, T, E_R, E_G, E_B)


def process_frame(frame_path, a: float, b: float, fps: float, output_dir):
    """Обработка одного кадра"""
    # Загрузка и усреднение данных по вертикали
    frame_data = np.load(frame_path)
    spectrum = np.mean(frame_data, axis=0)
    wavelengths = a * np.arange(spectrum.shape[0]) + b

    mask = (wavelengths >= 380) & (wavelengths <= 700)
    wavelengths = wavelengths[mask]
    spectrum = spectrum[mask, :]

    (SR, SG, SB, T, E_R, E_G, E_B) = scale()
    S_funcs = [SR, SG, SB]
    E_funcs = [E_R, E_G, E_B]  # Функции относительных погрешностей
    d_values = [dR, dG, dB]

    scaled_spec = np.zeros_like(spectrum)
    rel_err = np.zeros_like(spectrum)
    rel_err_img = np.zeros_like(spectrum)

    epsilon = np.finfo(float).eps * 1e3

    for i in range(3):
        # Вычисление чувствительности с учетом погрешности
        S = S_funcs[i](wavelengths)
        E_i = E_funcs[i](wavelengths)

        # Используем среднее значение для масштабирования
        denom = S * T(wavelengths)
        scaled_spec[:, i] = spectrum[:, i] / denom

        # Погрешность исходной интенсивности
        denom2 = spectrum[:, i]
        denom2 = np.clip(denom2, epsilon, None)
        rel_err_img[:, i] = abs(d_values[i] / denom2)  # Относительная погрешность исходной

        # Полная погрешность масштабированной интенсивности
        term1 = (d_values[i] / denom2) ** 2
        term2 = E_i ** 2  # Вклад погрешности чувствительности
        rel_err[:, i] = np.sqrt(term1 + term2)  # Суммирование в квадратуре

    frame_num = int(frame_path.stem.split("_")[1])
    time_ms = (frame_num / fps) * 1000

    result = {
        'wavelengths': wavelengths,
        'spectrum': spectrum,
        'rel_err_img': rel_err_img,
        'scaled_spec': scaled_spec,
        'rel_err': rel_err,
        'time_ms': time_ms,
        'original_shape': frame_data.shape
    }

    output_npy_path = output_dir / f"spectrum_{frame_num:02d}.npy"
    np.save(output_npy_path, result)

    data = {
        'time_us': 1e3 * time_ms,
        'frame_num': frame_num,
        'wavelength': wavelengths,
        'original_R': spectrum[:, 0],
        'original_G': spectrum[:, 1],
        'original_B': spectrum[:, 2],
        'eR': rel_err_img[:, 0],
        'eG': rel_err_img[:, 1],
        'eB': rel_err_img[:, 2],
        'scaled_R': scaled_spec[:, 0],
        'scaled_G': scaled_spec[:, 1],
        'scaled_B': scaled_spec[:, 2],
        'scaled_eR': rel_err[:, 0],
        'scaled_eG': rel_err[:, 1],
        'scaled_eB': rel_err[:, 2],
        'SR': S_funcs[0](wavelengths),
        'SG': S_funcs[1](wavelengths),
        'SB': S_funcs[2](wavelengths),
        'T': T(wavelengths)
    }

    df = pd.DataFrame(data)
    output_csv_path = output_dir / f"spectrum_{frame_num:02d}.csv"
    df.to_csv(output_csv_path, index=False, sep=';', decimal=',')

    # Вызов функции построения графиков
    plot_spectrum(data, frame_num, output_dir)

    return output_npy_path, output_csv_path, (frame_num, df)


def plot_spectrum(data, frame_num, output_dir):
    """Построение графиков для кадра"""
    channels = ['R', 'G', 'B']
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Цвета для графиков
    colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    alpha_fill = 0.3  # Прозрачность для заливки ошибок

    # ============= ЧЕРНО-БЕЛЫЕ ГРАФИКИ =============
    # Черно-белые стили линий и паттерны заливки
    line_styles = ['-', '--', '-.']
    hatch_patterns = ['///', '\\\\\\', '|||']
    gray_levels = [0.2, 0.4, 0.6]  # Уровни серого для заливки

    # График оригинальной интенсивности (ЧБ)
    plt.figure(figsize=(8, 4))
    for i, ch in enumerate(channels):
        y = data[f'original_{ch}']
        yerr = y * data[f'e{ch}']

        plt.plot(data['wavelength'], y,
                 label=ch,
                 linestyle=line_styles[i],
                 color='black',
                 linewidth=1.5)

        plt.fill_between(data['wavelength'],
                         y - yerr,
                         y + yerr,
                         alpha=0.7,
                         hatch=hatch_patterns[i],
                         edgecolor=f'{gray_levels[i]}',
                         facecolor='white'
                         )

    plt.xlabel('Длина волны, нм', fontsize=15)
    plt.ylabel('Интенсивность', fontsize=15)
    plt.title(f'Зарегистрированная интенсивность (ЧБ). Разряд 4. Кадр:{frame_num}', fontsize=15)
    plt.legend(fontsize=15)
    plt.xlim(400, 700)
    plt.ylim(-2000, 65534)
    plt.minorticks_on()
    plt.grid(True, axis='both', which='both', linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(plot_dir / f"spectrum_{frame_num:02d}_original_bw.png", dpi=300, bbox_inches='tight')
    plt.close()

    # График отмасштабированной интенсивности (ЧБ)
    plt.figure(figsize=(8, 4))
    for i, ch in enumerate(channels):
        y = data[f'scaled_{ch}']
        yerr = y * data[f'scaled_e{ch}']

        plt.plot(data['wavelength'], y,
                 label=ch,
                 linestyle=line_styles[i],
                 color='black',
                 linewidth=1.5)

        plt.fill_between(data['wavelength'],
                         y - yerr,
                         y + yerr,
                         alpha=0.7,
                         hatch=hatch_patterns[i],
                         edgecolor=f'{gray_levels[i]}',
                         linestyle=line_styles[i],
                         facecolor='white'
                         )

    plt.xlabel('Длина волны, нм', fontsize=15)
    plt.ylabel('Усл. ед.', fontsize=15)
    plt.title(f'Отмасштабированная интенсивность (ЧБ). Разряд 4. Кадр:{frame_num}', fontsize=15)
    plt.legend(fontsize=15)
    plt.xlim(400, 700)
    plt.ylim(0, 1e6)
    plt.minorticks_on()
    plt.grid(True, axis='both', which='both', linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(plot_dir / f"spectrum_{frame_num:02d}_scaled_bw.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============= ЦВЕТНЫЕ ГРАФИКИ =============
    # График оригинальной интенсивности (цветной)
    plt.figure(figsize=(12, 6))
    for ch in channels:
        y = data[f'original_{ch}']
        yerr = y * data[f'e{ch}']
        color = colors[ch]

        plt.plot(data['wavelength'], y,
                 label=ch,
                 color=color,
                 linewidth=1.5)

        plt.fill_between(data['wavelength'],
                         y - yerr,
                         y + yerr,
                         alpha=alpha_fill,
                         color=color)

    plt.xlabel('Длина волны, нм', fontsize=15)
    plt.ylabel('Интенсивность', fontsize=15)
    plt.title(f'Зарегистрированная интенсивность (цвет). Разряд 4. Кадр:{frame_num}', fontsize=15)
    plt.legend(fontsize=15)
    plt.xlim(400, 700)
    plt.ylim(-2000, 65534)
    plt.minorticks_on()
    plt.grid(True, axis='both', which='both', linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(plot_dir / f"spectrum_{frame_num:02d}_original_color.png", dpi=300, bbox_inches='tight')
    plt.close()

    # График отмасштабированной интенсивности (цветной)
    plt.figure(figsize=(12, 6))
    for ch in channels:
        y = data[f'scaled_{ch}']
        yerr = y * data[f'scaled_e{ch}']
        color = colors[ch]

        plt.plot(data['wavelength'], y,
                 label=ch,
                 color=color,
                 linewidth=1.5)

        plt.fill_between(data['wavelength'],
                         y - yerr,
                         y + yerr,
                         alpha=alpha_fill,
                         color=color)

    plt.xlabel('Длина волны, нм', fontsize=15)
    plt.ylabel('Усл. ед.', fontsize=15)
    plt.title(f'Отмасштабированная интенсивность (цвет). Разряд 4. Кадр:{frame_num}', fontsize=15)
    plt.legend(fontsize=15)
    plt.xlim(400, 700)
    plt.ylim(0, 1e6)
    plt.minorticks_on()
    plt.grid(True, axis='both', which='both', linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(plot_dir / f"spectrum_{frame_num:02d}_scaled_color.png", dpi=300, bbox_inches='tight')
    plt.close()


def process_discharge_folder(folder_path):
    """Обработка папки с данными разряда"""
    match = re.search(r'(\d+)$', folder_path.name)
    if not match:
        print(f"Invalid folder name: {folder_path.name}")
        return

    discharge_num = int(match.group(1))

    try:
        calibration = get_calibration(discharge_num)
        fps = FPS_CONFIG[str(discharge_num)]
    except (KeyError, ValueError) as e:
        print(f"Error getting calibration or FPS for discharge {discharge_num}: {e}")
        return

    a, b = 0.745677888989991, 266.7979981801634
    processed_dir = folder_path / "processed"
    preview_dir = folder_path / "previews"
    processed_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    frame_files = sorted(folder_path.glob("frame_*.npy"))
    all_frame_data = []

    for frame_path in frame_files:
        try:
            spectrum_path = process_frame(frame_path, a, b, fps, processed_dir)
            all_frame_data.append(spectrum_path[2])  # Добавляем (frame_num, df)
        except Exception as e:
            print(f"Error processing {frame_path.name}: {str(e)}")

    # Сохранение всех кадров в Excel
    excel_path = processed_dir / "all_frames.xlsx"
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        for frame_num, df in sorted(all_frame_data, key=lambda x: x[0]):
            sheet_name = f"Frame_{frame_num:02d}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == '__main__':
    # Обрабатываю только 4 разряд!!!!!!!____________
    for discharge_dir in OUTPUT_DIR.glob("[4]"):
        if discharge_dir.is_dir():
            print(f"Processing: {discharge_dir.name}")
            process_discharge_folder(discharge_dir)