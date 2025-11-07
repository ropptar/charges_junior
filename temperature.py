import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.constants import c
from pathlib import Path
from scipy.optimize import curve_fit

# Конфигурации
OUTPUT_DIR = Path(__file__).parent / "output"
LINE_PARAMS = {
    "Ha": {
        "transition": (3, 2),
        "lambda_nominal": 656.279,
        "A": 4.4101e7,
        "g": 18,
        "E_upper": -13.6 / (3 ** 2),
        "E_lower": -13.6 / (2 ** 2)
    },
    "Hb": {
        "transition": (4, 2),
        "lambda_nominal": 486.135,
        "A": 8.4193e6,
        "g": 32,
        "E_upper": -13.6 / (4 ** 2),
        "E_lower": -13.6 / (2 ** 2)
    },
    "Hg": {
        "transition": (5, 2),
        "lambda_nominal": 434.0472,
        "A": 2.5403e6,
        "g": 50,
        "E_upper": -13.6 / (5 ** 2),
        "E_lower": -13.6 / (2 ** 2)
    },
    "Hd": {
        "transition": (6, 2),
        "lambda_nominal": 410.1734,
        "A": 9.732e5,
        "g": 72,
        "E_upper": -13.6 / (6 ** 2),
        "E_lower": -13.6 / (2 ** 2)
    }
}
LINE_ORDER = ['Ha', 'Hb', 'Hg', 'Hd']
LINE_PAIRS = [
    ('Ha', 'Hb'),
    ('Ha', 'Hg'),
    ('Ha', 'Hd'),
    ('Hb', 'Hg'),
    ('Hb', 'Hd'),
    ('Hg', 'Hd')
]
# Постоянная Планка в эВ·с
h = 4.1e-15
# Скорость света в м/с
c = 3e8

def load_sensitivity(file_path):
    """Загрузка данных чувствительности детектора"""
    df = pd.read_csv(file_path)
    return {
        "R": interp1d(df["Wavelength"], df["SR"], kind='linear', fill_value='extrapolate'),
        "G": interp1d(df["Wavelength"], df["SG"], kind='linear', fill_value='extrapolate'),
        "B": interp1d(df["Wavelength"], df["SB"], kind='linear', fill_value='extrapolate')
    }


def load_transmission(file_path):
    """Загрузка данных спектра пропускания стекла"""
    df = pd.read_csv(file_path)
    return interp1d(df["Wavelength"], df["Total"], kind='linear', fill_value='extrapolate')


def calculate_temperature(I1, I2, line1, line2):
    """Расчет температуры для конкретной комбинации"""
    try:
        l1 = LINE_PARAMS[line1]
        l2 = LINE_PARAMS[line2]

        # Проверяем, чтобы I1 и I2 не были NaN
        if pd.isna(I1) or pd.isna(I2) or I2 == 0:
            return np.nan

        # Вычисляем отношение
        R = (I1 * l2['A'] * l2['g'] * l1['lambda_nominal']) / \
            (I2 * l1['A'] * l1['g'] * l2['lambda_nominal'])
        if R <= 1:
            return np.nan
        # Формула температуры (в эВ)
        T = ((l2['E_upper'] - l1['E_upper'])) / np.log(R)
        # print(R, T)
        if T > 10:
            return np.nan
        return round(T, 2) if not np.isinf(T) else np.nan

    except Exception as e:
        print(f'Ошибка расчета: {e}')
        return np.nan


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def plot_time_spectrum(fit_data, combined_data, i, time, fit_dir, channel_colors, line_styles):
    """
    Строит график для одного временного отсчёта, включая:
    - исходный спектр
    - отмасштабированный спектр
    - аппроксимацию гауссом для всех линий и каналов
    """
    plt.figure(figsize=(16, 6))

    # Исходный и отмасштабированный спектр
    wavelengths = combined_data['wavelengths']

    for ch in range(3):
        spectrum_interp_func = combined_data['spectrum_interp'][i][ch]
        scaled_interp_func = combined_data['scaled_spec_interp'][i][ch]

        plt.plot(wavelengths, spectrum_interp_func(wavelengths),
                 color=channel_colors[ch],
                 linestyle=line_styles['original'],
                 alpha=0.5,
                 label=f'Original (Ch {ch})')

        plt.plot(wavelengths, scaled_interp_func(wavelengths),
                 color=channel_colors[ch],
                 linestyle=line_styles['scaled'],
                 alpha=0.7,
                 label=f'Scaled (Ch {ch})')

    # Гауссовы аппроксимации
    for fit in fit_data:
        ch = fit['ch']
        wvl_fit = fit['wvl_fit']
        popt = fit['popt']
        plt.plot(wvl_fit, gaussian(wvl_fit, *popt),
                 color=channel_colors[ch],
                 linestyle=line_styles['fit'],
                 alpha=0.9,
                 linewidth = 3,
                 label=f'Gaussian Fit (Ch {ch})')

    # Настройка графика
    plt.title(f"Time: {time} μs")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    # plt.yscale('log')
    plt.ylim(1e-2, 1e10)

    plt.xlim(400, 700)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fit_dir / f"t{time:.1f}.png", dpi=300)
    plt.close()


def fit_gaussian_to_line(wvl_fit, scaled_interp_func, wvl_nominal, fit_range):
    """
    Аппроксимирует спектральную линию гауссовой функцией и возвращает площадь под кривой.
    Если аппроксимация не удалась, возвращается значение в центральной точке.
    """
    try:
        scaled_fit = scaled_interp_func(wvl_fit)
    except:
        scaled_fit = np.full_like(wvl_fit, np.nan)

    if np.isnan(scaled_fit).all():
        return np.nan, None

    A0 = np.max(scaled_fit)
    mu0 = wvl_nominal
    sigma0 = fit_range / 2.355  # Перевод FWHM в σ

    try:
        popt, _ = curve_fit(gaussian, wvl_fit, scaled_fit,
                            p0=[A0, mu0, sigma0], maxfev=1000)
        A_fit, mu_fit, sigma_fit = popt
        area = A_fit
        return area, popt
    except RuntimeError:
        return scaled_interp_func(wvl_nominal), None


def process_time_point(combined_data, i, time, fit_range, wvl_step):
    """
    Обрабатывает спектральные данные для одного временного отсчёта.
    Возвращает:
    - rows: список строк для DataFrame с area и значением в точке.
    - fit_data: данные для построения графиков.
    """
    rows = []
    fit_data = []
    for key in LINE_PARAMS:
        wvl_nominal = LINE_PARAMS[key]['lambda_nominal']
        for ch in range(3):
            wvl_fit = np.arange(wvl_nominal - fit_range / 2,
                                wvl_nominal + fit_range / 2 + wvl_step,
                                wvl_step)
            # Получаем значение в точке wvl_nominal
            spectrum_value = combined_data['spectrum_interp'][i][ch](wvl_nominal)
            scaled_value = combined_data['scaled_spec_interp'][i][ch](wvl_nominal)

            # Расчет площади (area) и параметров гауссианы
            area, popt = fit_gaussian_to_line(wvl_fit, combined_data['scaled_spec_interp'][i][ch],
                                              wvl_nominal, fit_range)

            # Сохранение результата
            row = {
                'time': time,
                'line': key,
                'channel': ch,
                'scaled_value': scaled_value,  # Значение в длине волны
                'area': area if area else None  # Используем area или значение
            }
            rows.append(row)

            # Данные для графика
            if popt is not None:
                fit_data.append({
                    'ch': ch,
                    'wvl_fit': wvl_fit,
                    'popt': popt
                })
    return rows, fit_data


def compute_ln_I(intensity, denom):
    """Вычисляет ln(I/g) с проверкой на ноль"""
    if pd.isna(intensity) or intensity <= 0:
        return np.nan
    return np.log(intensity / denom)


'''def add_boltzmann_columns(master_df):
    """Добавляет колонки ln(I/g) для всех каналов и линий"""
    for name in LINE_PARAMS.keys():
        for channel in ['CR', 'CG', 'CB']:
            col_name = f"ln_{channel}_{name}"
            master_df[col_name] = np.nan

    for (charge, time, name), row in master_df.iterrows():
        for channel in ['CR', 'CG', 'CB']:
            intensity = row[channel]
            g = LINE_PARAMS[name]['g']
            A = LINE_PARAMS[name]['A']
            w = 2 * np.pi * c / (LINE_PARAMS[name]['lambda_nominal']) * 1e9
            denom = g * A * w
            col_name = f"ln_{channel}_{name}"
            master_df.at[(charge, time, name), col_name] = compute_ln_I(intensity, denom)

    return master_df'''


'''def plot_boltzmann(group, group_name):
    """Строит график Болцмана для одной группы с линейной регрессией"""
    plt.figure(figsize=(13, 8))

    # Цвета для линий Ha, Hb, Hg, Hd
    color_map = {
        'Ha': 'red',  # Ha - красный
        'Hb': 'green',  # Hb - зелёный
        'Hg': 'blue',  # Hg - синий
        'Hd': 'purple'  # Hd - фиолетовый
    }

    # Маркеры для каналов CR, CG, CB
    marker_map = {
        'CR': 'o',  # CR - точка
        'CG': '^',  # CG - треугольник
        'CB': 's'  # CB - квадратик
    }

    # Списки для хранения данных
    E_upper_list = []
    ln_I_over_g_list = []

    # Для избежания дублирования подписей в легенде
    added_labels = set()

    # Перебираем все линии (Ha, Hb, Hg, Hd)
    for name in LINE_PARAMS.keys():
        try:
            row = group.xs(name, level='name')  # Получаем строку для текущей линии
        except KeyError:
            continue

        # Энергия верхнего уровня
        E_upper = LINE_PARAMS[name]['E_upper']

        # Перебираем все каналы (CR, CG, CB)
        for channel in ['CR', 'CG', 'CB']:
            col_name = f"ln_{channel}_{name}"  # Имя колонки с ln(I/g)
            ln_I_over_g = row[col_name].item()  # Значение ln(I/g)

            # Проверяем, что значение не NaN
            if not pd.isna(ln_I_over_g):
                # Добавляем данные для регрессии
                E_upper_list.append(E_upper)
                ln_I_over_g_list.append(ln_I_over_g)

                # Получаем цвет и маркер
                color = color_map.get(name, 'black')
                marker = marker_map.get(channel, 'o')

                # Формируем подпись
                label = f"{name} ({channel})"

                # Добавляем точку на график
                if label not in added_labels:
                    plt.scatter(E_upper, ln_I_over_g, color=color, marker=marker, label=label)
                    added_labels.add(label)
                else:
                    plt.scatter(E_upper, ln_I_over_g, color=color, marker=marker, label=None)

    # Проверяем, достаточно ли данных для регрессии
    if len(E_upper_list) >= 2:
        x = np.array(E_upper_list)
        y = np.array(ln_I_over_g_list)

        # Выполняем линейную регрессию
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        try:
            T = -1 / slope  # Рассчитываем температуру
        except ZeroDivisionError:
            T = np.nan
        # Строим регрессионную прямую
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'r-', label=f"Fit: T={T:.2f} eV, R²={r_value ** 2:.2f}")

    # Настраиваем график
    plt.xlabel('E_upper (eV)')
    plt.ylabel('ln(I/(g*A*w)')
    plt.title(f'Boltzmann Plot - {group_name}')
    plt.legend()
    plt.grid(True)

    # Сохраняем график
    plt.savefig(f"boltzmann_plot_{group_name}.png", dpi=300, bbox_inches='tight')
    plt.close()'''


'''def plot_temperature_groups(T_dict, dir_path, count):
    """Строит и сохраняет график температур с цветовой группировкой и разными маркерами"""
    save_dir = Path(dir_path).parent.parent / "temperatures"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Цвета для пар линий
    def get_color_group(pair):
        parts = pair.split('-')
        line1, line2 = parts[0], parts[1]
        if 'Ha' in {line1, line2}:
            return 'black'
        if sorted([line1, line2]) == ['Hb', 'Hg']:
            return 'red'
        if sorted([line1, line2]) == ['Hb', 'Hd']:
            return 'green'
        if sorted([line1, line2]) == ['Hg', 'Hd']:
            return 'blue'
        return 'gray'

    markers = {'scaled': 'o', 'area': '^'}

    # Обработка каждого типа данных
    for method, T_df in T_dict.items():
        df = T_df.reset_index().melt(
            id_vars='time', var_name='pair', value_name='temp'
        )
        df['color'] = df['pair'].apply(get_color_group)

        # Отображение данных
        for color in ['black', 'red', 'green', 'blue']:
            subset = df[(df['color'] == color) & (df['temp'].notna())]
            plt.scatter(
                subset['time'], subset['temp'],
                c=color, marker=markers[method],
                label=f"{method} ({color})", alpha=0.7, edgecolors='none'
            )

    # Настройка графика
    plt.title(f'Динамика температуры разряда {count}')
    plt.xlabel('Время, мкс')
    plt.ylabel('Температура, эВ')
    plt.minorticks_on()
    plt.grid(True, axis='both')

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Метод scaled',
               markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Метод area',
               markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Ha-*',
               markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Hb-Hg',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Hb-Hd',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Hg-Hd',
               markerfacecolor='blue', markersize=10)
    ]
    plt.legend(handles=legend_elements, title='Методы и группы')

    # Сохранение графика
    save_path = save_dir / f"temperature_plot_{count}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()'''


'''def prepare_data_for_regression(group):
    """Подготавливает данные для линейной регрессии"""
    E_upper_list = []
    ln_I_over_g_list = []

    for name in LINE_PARAMS.keys():
        try:
            row = group.xs(name, level='name')
        except KeyError:
            continue

        E_upper = LINE_PARAMS[name]['E_upper']

        for channel in ['CR', 'CG', 'CB']:
            col_name = f"ln_{channel}_{name}"
            ln_I_over_g = row[col_name].item()

            if not pd.isna(ln_I_over_g):
                E_upper_list.append(E_upper)
                ln_I_over_g_list.append(ln_I_over_g)

    return np.array(E_upper_list), np.array(ln_I_over_g_list)'''


'''def perform_regression_and_calculate_temp(group):
    """Выполняет линейную регрессию и возвращает температуру и её стандартную ошибку"""
    x, y = prepare_data_for_regression(group)

    if len(x) < 2:
        return np.nan, np.nan  # Недостаточно данных для регрессии

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    T = -1 / (slope)
    T_std_err = abs(-1 / (slope + std_err) - T)  # Приближенная погрешность

    return T, T_std_err
'''

def planck_model(lam, T, A):
    """
    Планковская функция для длины волны λ (нм) и температуры T (эВ).
    A - масштабный коэффициент.
    """
    nu = c*1e9/lam
    denominator = np.exp(h * nu /T) - 1
    return A * (2 * h * nu**3 * c**2) / denominator


def fit_planck_spectrum(wavelengths, scaled_interp_func, fit_ranges, initial_guess=[1.0, 1e-10]):
    """
    Подгоняет Планковскую кривую к данным в заданных диапазонах.

    Параметры:
        wavelengths (np.array): Длины волн.
        scaled_interp_func (callable): Интерполяционная функция спектра.
        fit_ranges (list): Диапазоны [(start, end), ...] для подгонки.
        initial_guess (list): Начальные значения [T, A].

    Возвращает:
        T_fit (float): Подогнанная температура (эВ).
        A_fit (float): Масштабный коэффициент.
        x_data, y_data (np.array): Данные для графика.
    """
    x_data = []
    y_data = []

    # Сбор данных из указанных диапазонов
    for start, end in fit_ranges:
        mask = (wavelengths >= start) & (wavelengths <= end)
        x_peak = wavelengths[mask]
        y_peak = scaled_interp_func(x_peak)
        x_data.extend(x_peak)
        y_data.extend(y_peak)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Проверка на пустые данные
    if len(x_data) == 0 or np.isnan(y_data).all():
        return None, None, x_data, y_data

    # Подгонка
    try:
        popt, _ = curve_fit(
            planck_model,
            x_data, y_data,
            p0=initial_guess,
            maxfev=1000,
            bounds=(0, [np.inf, 1e-5])  # Ограничения на параметры
        )
        T_fit, A_fit = popt
        return T_fit, A_fit, x_data, y_data
    except RuntimeError as e:
        print(f"Ошибка подгонки: {e}")
        return None, None, x_data, y_data


def plot_planck_fit(wavelengths, scaled_interp_func, T_fit, A_fit, x_data, y_data, fit_ranges, channel, time):
    """
    Строит график спектра и аппроксимации Планка.
    """
    plt.figure(figsize=(12, 6))

    # Исходный спектр
    plt.plot(wavelengths, scaled_interp_func(wavelengths), label='Отмасштабированный спектр', color='black')

    # Диапазоны подгонки
    for start, end in fit_ranges:
        plt.axvspan(start, end, color='gray', alpha=0.2, label='Диапазон подгонки')

    # Данные подгонки
    plt.scatter(x_data, y_data, color='blue', label='Данные для подгонки', zorder=5)

    # Аппроксимация Планка
    if T_fit is not None and A_fit is not None:
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = planck_model(x_fit, T_fit, A_fit)
        plt.plot(x_fit, y_fit, 'r--', label=f'Планковская кривая (T={T_fit:.2f} эВ)', linewidth=2)

    plt.title(f'Канал {channel}, Время {time} мкс')
    plt.xlabel('Длина волны (нм)')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"planck_fit_plots/channel_{channel}_time_{time}.png", dpi=300)
    plt.close()
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_temperature_dynamics_unified(result_df, dir_path, count):
    """
    Строит графики динамики температуры по времени для каждой пары линий,
    отображая все каналы на одном графике. Использует:
    - разные цвета для ch1 (0 → r, 1 → g, 2 → b)
    - разные формы маркеров для ch2 (0 → o, 1 → s, 2 → ^)
    - разную прозрачность (alpha) для методов: scaled (0.6), area (1.0)
    - в легенде отображаются пары в формате r-g, b-b и т.д.
    """
    save_dir = Path(dir_path).parent.parent / "temperatures"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Цвета и маркеры
    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    marker_map = {0: 'o', 1: 's', 2: '^'}
    alpha_map = {'scaled': 0.6, 'area': 1.0}

    # Сопоставление числовых индексов с буквами
    channel_label_map = {'0': 'r', '1': 'g', '2': 'b'}

    # Уникальные пары линий
    line_pairs = result_df.columns.get_level_values('line_pair').unique()

    for line_pair in line_pairs:
        plt.figure(figsize=(12, 7))
        legend_elements = set()  # Для уникальных записей

        for col in result_df.columns:
            method, current_line_pair, channel = col
            if current_line_pair != line_pair:
                continue

            try:
                # Разбиваем канал на ch1 и ch2
                ch1, ch2 = map(int, channel.split('-'))

                # Формируем обозначение канала в виде r-g, b-b и т.д.
                transformed_channel = f"{channel_label_map[str(ch1)]}-{channel_label_map[str(ch2)]}"

                # Цвет, маркер и прозрачность
                color = color_map[ch1]
                marker = marker_map[ch2]
                alpha = alpha_map[method]

                # Получаем данные
                data = result_df[col]

                # Пропускаем, если все NaN
                if data.isna().all():
                    continue

                # Рисуем точку
                plt.scatter(
                    result_df.index, data,
                    color=color, alpha=alpha,
                    marker=marker, s=60, edgecolor='black', linewidth=0.5,
                    label=f"{transformed_channel} ({method})"
                )

                # Добавляем элемент в легенду (только один раз на комбинацию)
                legend_key = (ch1, ch2, method)
                legend_label = f"{transformed_channel} ({method})"

                # Проверяем, есть ли уже такой элемент в легенде
                existing_labels = [handle.get_label() for handle in legend_elements]
                if legend_label not in existing_labels:
                    legend_elements.add(
                        Line2D([0], [0], color=color, lw=2, marker=marker,
                               markersize=8, alpha=alpha, label=legend_label)
                    )

            except Exception as e:
                print(f"Ошибка при обработке {line_pair}, {channel}: {e}")

        # Оформление графика
        plt.title(f'Разряд {count}. Динамика температуры для пары линий: {line_pair}')
        plt.xlabel('Время (мкс)')
        plt.ylabel('Температура')
        plt.grid(True)

        # Легенда
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # Сохранение графика
        file_name = f"charge_{count}_{line_pair}_combined.png"
        plt.tight_layout()
        plt.savefig(save_dir / file_name, dpi=150, bbox_inches='tight')
        plt.close()

def load_and_combine_spectra(output_dir):
    # Список для хранения данных
    all_data = {
        'wavelengths': None,
        'spectrum_interp': [],
        'sens_spec_interp': [],
        'trans_spec_interp': [],
        'scaled_spec_interp': [],
        'time_us': []
    }

    # Поиск всех .npy файлов в папке
    npy_files = sorted(output_dir.glob("spectrum_*.npy"))

    for file_path in npy_files:
        # Чтение файла
        result = np.load(file_path, allow_pickle=True).item()

        wavelengths = result['wavelengths']
        mask = (wavelengths >= 380) & (wavelengths <= 670)
        filtered_wavelengths = wavelengths[mask]
        all_data['wavelengths'] = filtered_wavelengths


        # Создание интерполяционных функций для spectrum
        spectrum_data = result['spectrum']
        spectrum_interps = [interp1d(result['wavelengths'], spectrum_data[:, i],
                                     kind='linear', bounds_error=False,
                                     fill_value="extrapolate") for i in range(3)]
        all_data['spectrum_interp'].append(spectrum_interps)

        # Создание интерполяционных функций для sens_spec
        sens_spec_data = result['sens_spec']
        sens_spec_interps = [interp1d(result['wavelengths'], sens_spec_data[:, i],
                                      kind='linear', bounds_error=False,
                                      fill_value="extrapolate") for i in range(3)]
        all_data['sens_spec_interp'].append(sens_spec_interps)

        # Создание интерполяционных функций для trans_spec
        trans_spec_data = result['trans_spec']
        trans_spec_interps = [interp1d(result['wavelengths'], trans_spec_data[:, i],
                                       kind='linear', bounds_error=False,
                                       fill_value="extrapolate") for i in range(3)]
        all_data['trans_spec_interp'].append(trans_spec_interps)

        # Создание интерполяционных функций для scaled_spec
        scaled_spec_data = result['scaled_spec']
        scaled_spec_interps = [interp1d(result['wavelengths'], scaled_spec_data[:, i],
                                        kind='linear', bounds_error=False,
                                        fill_value="extrapolate") for i in range(3)]
        all_data['scaled_spec_interp'].append(scaled_spec_interps)

        # Добавление времени в микросекундах
        all_data['time_us'].append(1e3 * result['time_ms'])

    # Преобразование временных меток в массив NumPy
    all_data['time_us'] = np.array(all_data['time_us'])

    return all_data


def process_charge(dir_path, plot_fits=True):
    """Обрабатывает заряды и возвращает два DataFrame для методов scaled и area"""
    combined_data = load_and_combine_spectra(dir_path)
    rows = []
    fit_range = 5.0
    wvl_step = 0.1

    # Цвета и стили для графиков
    channel_colors = {0: 'red', 1: 'green', 2: 'blue'}
    line_styles = {'original': '-', 'scaled': '--', 'fit': ':'}

    # Папка для графиков
    if plot_fits:
        fit_dir = Path(dir_path) / "fit_plots"
        fit_dir.mkdir(exist_ok=True)

    # Обработка временных точек
    for i, time in enumerate(combined_data['time_us']):
        time_rows, fit_data = process_time_point(
            combined_data, i, time, fit_range, wvl_step
        )
        rows.extend(time_rows)

        if plot_fits:
            plot_time_spectrum(fit_data, combined_data, i, time,
                                fit_dir, channel_colors, line_styles)

    df = pd.DataFrame(rows)
    print(combined_data)
    # Уникальные временные точки
    times = df['time'].unique()

    # Группы методов
    groups = ['scaled', 'area']

    # Генерация всех комбинаций каналов (0-0, 0-1, ..., 2-2)
    channel_combinations = [(ch1, ch2) for ch1 in [0, 1, 2] for ch2 in [0, 1, 2]]

    # Подготовка данных для нового DataFrame
    result_data = []
    for time in times:
        time_df = df[df['time'] == time]
        data_map = {}
        for _, row in time_df.iterrows():
            key = (row['line'], row['channel'])
            data_map[key] = {
                'scaled': row['scaled_value'],
                'area': row['area']
            }

        temp_record = {'time': time}
        for group in groups:
            for line1, line2 in LINE_PAIRS:
                pair_name = f"{line1}-{line2}"
                for ch1, ch2 in channel_combinations:
                    val1 = data_map.get((line1, ch1), {}).get(group, None)
                    val2 = data_map.get((line2, ch2), {}).get(group, None)

                    if val1 is not None and val2 is not None:
                        temp = calculate_temperature(val1, val2, line1, line2)
                    else:
                        temp = None
                    col_key = (group, pair_name, f"{ch1}-{ch2}")
                    temp_record[col_key] = temp
        result_data.append(temp_record)

    # Создание DataFrame с мультииндексом
    result_df = pd.DataFrame(result_data).set_index('time')
    result_df.columns = pd.MultiIndex.from_tuples(
        result_df.columns,
        names=['method', 'line_pair', 'channels']
    )
    return result_df


def analyze_planck_spectrum(combined_data, output_dir, plot_results=True):
    """
    Анализирует спектральные данные методом Планковской аппроксимации

    Args:
        combined_data: данные из load_and_combine_spectra
        output_dir: директория для сохранения графиков
        plot_results: флаг сохранения графиков

    Returns:
        results: словарь с результатами температурной аппроксимации
    """
    FWHM = 6
    # Создаем директорию для графиков Планковской аппроксимации
    planck_dir = Path(output_dir) / "planck_fits"
    planck_dir.mkdir(exist_ok=True)

    fit_ranges = [(params['lambda_nominal'] - FWHM/2, params['lambda_nominal'] +FWHM/2)
                  for params in LINE_PARAMS.values()]

    results = []

    # Цвета для каналов
    channel_colors = {0: 'red', 1: 'green', 2: 'blue'}

    # Обрабатываем каждый временной отсчет
    for i, time in enumerate(combined_data['time_us']):
        time_result = {'time': time}

        # Обрабатываем каждый цветовой канал
        for ch in range(3):
            # Получаем интерполяционную функцию для текущего канала
            scaled_interp_func = combined_data['scaled_spec_interp'][i][ch]

            # Подгоняем Планковскую кривую
            T_fit, A_fit, x_data, y_data = fit_planck_spectrum(
                combined_data['wavelengths'],
                scaled_interp_func,
                fit_ranges
            )

            # Сохраняем результаты
            time_result[f'channel_{ch}'] = {
                'temperature': T_fit,
                'amplitude': A_fit
            }

            # Строим график если требуется
            if plot_results:
                plt.figure(figsize=(12, 6))

                # Исходный спектр
                plt.plot(combined_data['wavelengths'],
                         scaled_interp_func(combined_data['wavelengths']),
                         label='Отмасштабированный спектр', color='black')

                # Диапазоны подгонки
                for start, end in fit_ranges:
                    plt.axvspan(start, end, color='gray', alpha=0.2)

                # Данные подгонки
                plt.scatter(x_data, y_data, color='blue',
                            label='Данные для подгонки', zorder=5)

                # Аппроксимация Планка
                if T_fit is not None and A_fit is not None:
                    x_fit = np.linspace(min(x_data), max(x_data), 100)
                    y_fit = planck_model(x_fit, T_fit, A_fit)
                    plt.plot(x_fit, y_fit, 'r--',
                             label=f'Планковская кривая (T={T_fit:.2f} эВ)',
                             linewidth=2)

                plt.title(f'Канал {channel_colors[ch].upper()}, Время {time} мкс')
                plt.xlabel('Длина волны (нм)')
                plt.ylabel('Интенсивность')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # Сохраняем график
                plt.savefig(planck_dir / f"t{time:.1f}_ch{ch}.png", dpi=300)
                plt.close()

        results.append(time_result)

    return results


def main():
    print("Start..._______________")
    count =1
    for discharge_dir in OUTPUT_DIR.glob("[1-4]"):
        if discharge_dir.is_dir():
            print(f"Processing: {discharge_dir.name}")
            folder_path = discharge_dir / "processed"
            T = process_charge(folder_path)
            plot_temperature_dynamics_unified(T, discharge_dir, count)
            combined_data = load_and_combine_spectra(folder_path)
            planck_results = analyze_planck_spectrum(combined_data, discharge_dir)
            count+=1
            # spectrum_files = sorted(folder_path.glob("spectrum_*.npy"))


if __name__ == "__main__":
    main()
