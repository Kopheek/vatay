'''
LR1p2.py - Лабораторная работа №1, пункт 2 "Распределение (псевдо)случайных чисел с одинаковым амплитудным спектром"

Выполнил студент 
гр. М3О-401Б-22
Сухинин В.М.
'''

import numpy as np # Устанавливаем стандартную библиотеку Python для работы с математическим анализом и линейной алгеброй
import matplotlib.pyplot as plot # Устанавливаем стандартную библиотеку для работы с графиками

# Параметры сигнала
count = 100 # количество точек
fs = 1000 # частота дискретизации, Гц

# Генерация псевдослучайного сигнала с одинаковым амплитудным спектром
random_phases = np.exp(1j * 2 * np.pi * np.random.rand(count//2-1)) # cоздаем случайные фазы
# Формируем комплексный спектр с одинаковой амплитудой
amplitude = 1
spectrum = np.zeros(count, dtype=complex)
spectrum[1:count//2] = amplitude * random_phases
spectrum[count//2+1:] = np.conj(spectrum[1:count//2][::-1])
spectrum[0] = 0
spectrum[count//2] = amplitude  # Найквист
# Обратное преобразование Фурье для получения временного сигнала

signal = np.fft.ifft(spectrum).real

# 2. Нормализация и квантизация в диапазоне [0,1] с шагом 0.1
signal = (signal - signal.min()) / (signal.max() - signal.min())  # нормируем в [0,1]
signal = np.round(signal * 10) / 10  # квантизация с шагом 0.1

# Гистограмма распределения значений
plot.figure(figsize=(12, 4))
plot.subplot(1, 3, 1)
plot.hist(signal, bins=30, edgecolor='black', alpha=0.7)
plot.xlabel('Значение')
plot.ylabel('Частота')
plot.title('Гистограмма распределения\nпри %s точках' % count)

# График процесса
t = np.arange(count)/fs
plot.subplot(1, 3, 2)
plot.plot(t, signal)
plot.xlabel('t, c')
plot.ylabel('Амплитуда')
plot.title('Временной процесс')

# Амплитудный спектр
freq = np.fft.fftfreq(count, d=1/fs)
fft_result = np.fft.fft(signal) # преобразование Фурье
amplitude_spectrum = np.abs(fft_result) # амплитудный спектр

plot.subplot(1, 3, 3)
plot.stem(freq[1:count//2], amplitude_spectrum[1:count//2], basefmt=" ")
plot.xlabel('Частота, Гц')
plot.ylabel('Амплитуда')
plot.title('Амплитудный спектр')

plot.tight_layout()
plot.show()