'''
LR1p31.py - Лабораторная работа №1, пункт 3 "Нормальное (Гауссово) случайное распределение при помощи метода rand() по центральной предельной теореме"

Выполнил студент 
гр. М3О-401Б-22
Сухинин В.М.
'''

import numpy as np # Устанавливаем стандартную библиотеку Python для работы с математическим анализом и линейной алгеброй
import matplotlib.pyplot as plot # Устанавливаем стандартную библиотеку для работы с графиками

# Параметры 
count = 100 # количество сгенерированных нормальных случайных чисел
M = 12 # количество слагаемых для ЦПТ (обычно 12 для хорошей аппроксимации)

# Генерация нормального распределения через ЦПТ
uniform_samples = np.random.rand(count, M) # Берем сумму M равномерных случайных чисел [0,1] и нормируем
gaussian_samples = np.sum(uniform_samples, axis=1) - M/2  # центрируем вокруг 0

# --- Нормализация в диапазон [0, 1] ---
gaussian_norm = (gaussian_samples - np.min(gaussian_samples)) / (np.max(gaussian_samples) - np.min(gaussian_samples))
gaussian_discrete = np.round(gaussian_norm / 0.1) * 0.1  # дискретизация по шагу 0.1
gaussian_discrete = np.clip(gaussian_discrete, 0, 1)  # защита от выхода за границы

# Гистограмма распределения
plot.subplot(1, 3, 1)
plot.hist(gaussian_discrete, bins=np.arange(0, 1.1, 0.1), density=True, alpha=0.7, edgecolor='black') # type: ignore
plot.title("Гистограмма нормального распределения\n(центральная предельная теорема, %s точек)" % count)
plot.xlabel("Значение")
plot.ylabel("Частота")
plot.grid(True)

# График процесса (значение vs время)
plot.subplot(1, 3, 2)
plot.plot(gaussian_samples)
plot.title("Процесс")
plot.xlabel("t, с")
plot.ylabel("Значение")
plot.grid(True)

# Амплитудный спектр
fft_vals = np.fft.fft(gaussian_samples) # преобразование Фурье
fft_freq = np.fft.fftfreq(len(gaussian_samples)) # type: ignore
fft_ampl = np.abs(fft_vals) # амплитудный спектр

plot.subplot(1, 3, 3)
plot.stem(fft_freq[:count//2], fft_ampl[:count//2], basefmt=" ")  # отображаем только положительные частоты
plot.title("Амплитудный спектр")
plot.xlabel("Частота")
plot.ylabel("Амплитуда")
plot.grid(True)

plot.tight_layout()
plot.show()