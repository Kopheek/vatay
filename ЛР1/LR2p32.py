'''
LR1p31.py - Лабораторная работа №1, пункт 3 "Нормальное (Гауссово) случайное распределение при помощи метода при помощи метода rand() по Брюссу-Мюллеру"

Выполнил студент 
гр. М3О-401Б-22
Сухинин В.М.
'''

import numpy as np # Устанавливаем стандартную библиотеку Python для работы с математическим анализом и линейной алгеброй
import matplotlib.pyplot as plot # Устанавливаем стандартную библиотеку для работы с графиками

# Параметры
count = 100 # Количество точек
mu = 0 # Среднее значение
sigma = 1 # Стандартное отклонение

# Генерация нормального распределения методом Брюсса-Мюллера
u1 = np.random.rand(count//2)
u2 = np.random.rand(count//2)

z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

x = np.concatenate((z0, z1))
x = mu + sigma * x  # Масштабируем под нужное среднее и стандартное отклонение

# Ограничение значений в [0,1] и округление до шага 0.1
x = np.clip(x, 0, 1)
x = np.round(x / 0.1) * 0.1

# --- Гистограмма распределения ---
plot.subplot(1, 3, 1)
plot.hist(x, bins=50, density=True, edgecolor='black', alpha=0.7)
plot.title("Гистограмма нормального распределения\n(Брюсс-Мюллер, %s точек)" % count)
plot.xlabel("Значение")
plot.ylabel("Частота")

# --- График процесса (значение vs время) ---
plot.subplot(1, 3, 2)
plot.plot(x)
plot.title('Процесс')
plot.xlabel("t, c")
plot.ylabel("Значение")

# --- Амплитудный спектр ---
X_f = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), d=1)  # d=1 — шаг времени
amplitude = np.abs(X_f)

plot.subplot(1, 3, 3)
plot.stem(freqs[1:count//2], amplitude[1:count//2], basefmt=" ")  # только положительные частоты
plot.title("Амплитудный спектр")
plot.xlabel("Частота")
plot.ylabel("Амплитуда")

plot.tight_layout()
plot.show()