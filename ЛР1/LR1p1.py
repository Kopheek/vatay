# -*- coding: utf-8 -*-

'''
LR1p1.py - Лабораторная работа №1, пункт 1 "Реализация равномерного псевдослучайного распределения методом "Лото"

Выполнил студент 
гр. М3О-401Б-22
Сухинин В.М.
'''

import random # Устанавливаем стандартную библиотеку Python для работы с (псево)случайностью
import matplotlib.pyplot as plot # Устанавливаем стандартную библиотеку для работы с графиками
from collections import Counter # Из стандартной библиотеки Python для работы с контейнерами устанавливает функцию для счёта частот
import numpy as np # Устанавливаем стандартную библиотеку Python для работы с математическим анализом и линейной алгеброй

# - 1. Реализация равномерного псевдослучайного распределения методом "Лото" - #

# Реализация метода "лото"
sequence = [i/10 for i in range(0, 11, 1)] # создаём список значений для выбора одного случайного
count = 100 # задаём количество запусков

results = [] # Задаём список для записи результатов запусков
for i in range(count): # Цикл для всех запусков
    results.append(random.choice(sequence)) # Записываем каждый результат

# Подсчёт частот
counts = Counter(results) # рассчёт частот функцией стандартной библиотеки Python
x = list(counts.keys()) # создаём список 
y = [counts[k] for k in x] # создаём список частот

# Построение гистограммы распределения
plot.subplot(1, 3, 1)
plot.hist(results, bins=len(sequence), edgecolor='black', alpha=0.7)
plot.title("Гистограмма распределения\nпри %s точках" % (count)) 
plot.xlabel("Значение")
plot.ylabel("Частота")
plot.grid(axis='y', linestyle='--', alpha=0.6)
#plot.show()

# Построение процесса
plot.subplot(1, 3, 2)
plot.plot(sequence)
plot.title('Процесс')
plot.xlabel('t, c')
plot.ylabel('Величина')
plot.grid(True, alpha=0.3)
#plot.show()

# Построение частотной хар-ки
fft_result = np.fft.fft(sequence) # преобразование Фурье
amplitude = np.abs(fft_result) # амплитудный спектр
freq = np.fft.fftfreq(len(sequence)) # type: ignore

# Строим только амплитудный спектр
plot.subplot(1, 3, 3)
plot.stem(freq[1:(len(freq)//2)], amplitude[1:(len(amplitude)//2)], basefmt=" ")
plot.title('Амплитудный спектр')
plot.xlabel('Частота')
plot.ylabel('Амплитуда')
plot.grid(True, alpha=0.3)
#plot.show()

plot.tight_layout()
plot.show()

# - 2. Распределение (псевдо)случайных чисел с одинаковым амплитудным спектром - #

