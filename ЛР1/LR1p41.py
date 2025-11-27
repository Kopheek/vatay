'''
LR1p41.py - Лабораторная работа №1, пункт 4 "Экспоненциальное распределение при помощи метода «лото»"

Выполнил студент 
гр. М3О-401Б-22
Кучеров И.В.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, shapiro


def lottery_uniform(n_samples, start=1, end=100):
    """
    Генерация равномерного распределения методом лото.
    """
    numbers_pool = list(range(start, end + 1))
    uniform_samples = []
   
    for _ in range(n_samples):
        np.random.shuffle(numbers_pool)
        number = numbers_pool[0]
        normalized_number = (number - start) / (end - start)
        uniform_samples.append(normalized_number)
   
    return np.array(uniform_samples)


def uniform_to_exponential(uniform_data, lambd=1.0):
    """
    Преобразование равномерного распределения в экспоненциальное.
   
    Args:
        uniform_data: массив из Uniform(0,1)
        lambd: параметр экспоненциального распределения
   
    Returns:
        Массив с экспоненциальным распределением
    """
    # Избегаем значений слишком близких к 0 и 1
    epsilon = 1e-10
    uniform_data = np.clip(uniform_data, epsilon, 1 - epsilon)
   
    # Метод обратного преобразования
    exponential_data = -np.log(1 - uniform_data) / lambd
    return exponential_data


def lottery_to_exponential(n_samples, lambd=1.0, lottery_size=100):
    """
    Получение экспоненциального распределения методом лото.
    """
    # Генерируем равномерное распределение методом лото
    uniform_data = lottery_uniform(n_samples, 1, lottery_size)
   
    # Преобразуем в экспоненциальное
    exponential_data = uniform_to_exponential(uniform_data, lambd)
   
    return exponential_data


# Параметры
n_samples = 10000
lambd = 1.0  # Параметр экспоненциального распределения


# Генерируем данные
exponential_data = lottery_to_exponential(n_samples, lambd)


print(f"Всего точек: {len(exponential_data)}")
print(f"Точек после фильтрации: {len(exponential_data)}")
print(f"Максимальное значение: {exponential_data.max():.2f}")
print(f"Процент отфильтрованных точек: {(1 - len(exponential_data)/len(exponential_data))*100:.2f}%")


# Построение графиков
plt.figure(figsize=(15, 5))


# 1. Исходное равномерное распределение (метод лото)
plt.subplot(1, 3, 1)
uniform_data = lottery_uniform(n_samples, 1, 100)
plt.hist(uniform_data, bins=20, alpha=0.7, edgecolor='black')
plt.title('Равномерное распределение\n(метод лото)')
plt.xlabel('Значение')
plt.ylabel('Плотность вероятности')
plt.grid(True, alpha=0.3)


# 2. Экспоненциальное распределение
plt.subplot(1, 3, 2)
plt.hist(exponential_data, bins=20, density=True, alpha=0.7, edgecolor='black')


# Теоретическая экспоненциальная кривая
x = np.linspace(0, 30, 100)
y = expon.pdf(x, scale=1/lambd)  # scale = 1/λ
plt.plot(x, y, 'r-', linewidth=2, label=f'Exp(λ={lambd})')


plt.title('Экспоненциальное распределение\n(из метода лото)')
plt.xlabel('Значение')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True, alpha=0.3)


# 3. Процесс во времени
plt.subplot(1, 3, 3)
plt.plot(exponential_data[:200])  # Первые 200 точек
plt.title('Процесс во времени\n(экспоненциальное распределение)')
plt.xlabel('Время (отсчеты)')
plt.ylabel('Величина')
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()


# Частотная характеристика
plt.figure(figsize=(10, 4))


fft_result = np.fft.fft(exponential_data)
freq = np.fft.fftfreq(len(exponential_data))
amplitude = np.abs(fft_result)
positive_mask = (freq > 0)
plt.stem(freq[positive_mask], amplitude[positive_mask], basefmt=" ")
plt.title('Частотная характеристика\n(экспоненциальное распределение)')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()
