import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла без заголовков
try:
    df = pd.read_csv('transmission.csv')

    # Проверка наличия нужных столбцов
    if 'Wavelength' not in df.columns or 'Total' not in df.columns:
        raise ValueError("В файле отсутствуют необходимые столбцы: 'Wavelength' или 'Total'")

    # Преобразование значений в числовые типы
    df['Wavelength'] = df['Wavelength'].str.replace(',', '.').astype(float)
    df['Total'] = df['Total'].str.replace(',', '.').astype(float)

    # Удаление строк с некорректными данными (NaN)
    df.dropna(inplace=True)

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(df['Wavelength'], df['Total'], color ='black', linestyle='-', label='Avantes')

    # Оформление графика
    plt.title('Спектр пропускания стекла в эксперименте')
    plt.xlabel('Длина волны, нм')
    plt.ylabel('Коэффициент пропускания')
    plt.legend()
    plt.grid(True, which='both', axis= 'both')
    plt.tight_layout()
    plt.ylim(0,1)
    plt.xlim(380, 700)
    # Отображение графика
    plt.show()

except FileNotFoundError:
    print("Ошибка: Файл 'transmission.csv' не найден.")
except Exception as e:
    print(f"Произошла ошибка: {e}")