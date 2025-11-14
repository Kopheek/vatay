'''
LR1p33.py - Лабораторная работа №1, пункт 3 "Нормальное (Гауссово) случайное распределение при помощи метода «лото»"

Выполнил студент 
гр. М3О-401Б-22
Сухинин В.М.
'''

import numpy as np # Устанавливаем стандартную библиотеку Python для работы с математическим анализом и линейной алгеброй
import matplotlib.pyplot as plot # Устанавливаем стандартную библиотеку для работы с графиками

# Параметры
count = 100 # количество случайных чисел
M = 12 # количество "билетов" для метода лото

# 1. Генерация нормального распределения методом "лото"
uniform_randoms = np.random.rand(count, M)
gaussian_randoms = np.sum(uniform_randoms, axis=1) - M/2
gaussian_randoms = gaussian_randoms / np.std(gaussian_randoms)

# 2. Нормализация в диапазон [0, 1]
min_val = np.min(gaussian_randoms)
max_val = np.max(gaussian_randoms)
normalized = (gaussian_randoms - min_val) / (max_val - min_val)

# 3. Дискретизация с шагом 0.1
discrete = np.round(normalized * 10) / 10  # округляем к ближайшему 0.1

# --- Гистограмма распределения ---
plot.subplot(1, 3, 1)
plot.hist(discrete, bins=np.arange(-0.05, 1.05, 0.1), density=True, edgecolor='black', alpha=0.7) # type: ignore
plot.title('Гистограмма нормального распределения\n(метод "лото", %s точек)' % count)
plot.xlabel('Значение')
plot.ylabel('Частота')
plot.grid(True)

# --- График процесса ---
plot.subplot(1, 3, 2)
plot.plot(discrete)
plot.title('Процесс')
plot.xlabel('Время')
plot.ylabel('Значение')
plot.grid(True)

# --- Амплитудный спектр ---
X_f = np.fft.fft(discrete)
freq = np.fft.fftfreq(len(discrete), d=1)  # d=1 — шаг времени
amplitude = np.abs(X_f)

plot.subplot(1, 3, 3)
plot.plot(freq[1:count//2], amplitude[1:count//2])
plot.title('Амплитудный спектр')
plot.xlabel('Частота')
plot.ylabel('Амплитуда')
plot.grid(True)

plot.tight_layout()
plot.show()