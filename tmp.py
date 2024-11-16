# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle


N = 1000
a = 100
bins = 10

# # Генерация данных
# radiuses = np.random.uniform(0, (a / (2 * np.pi)), size=N)
# angles = np.random.uniform(0, 2 * np.pi, size=N)
# x = radiuses * np.cos(angles)
# y = radiuses * np.sin(angles)

# # Создание графика
# plt.figure(figsize=(8, 8))

# # Построение рассеянного графика
# plt.scatter(x, y, c='purple')

# # Определение максимального радиуса
# rmax = np.max(radiuses)

# # Построение окружностей
# for i in range(1, bins + 1):
#     circle = Circle((0, 0), (i / bins) * rmax, fill=False)
#     plt.gca().add_artist(circle)

# # Настройка графика
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlim(-rmax, rmax)
# plt.ylim(-rmax, rmax)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot with Circles')
# plt.grid(True)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def density(r, a, N):
    return N / (a * r)

# Генерация данных
radiuses = np.random.uniform(0, (a / (2 * np.pi)), size=N)

# Вычисление плотности распределения
bins = 20
hist, bin_edges = np.histogram(radiuses, bins=bins, density=True)

# Вычисление теоретической плотности
x = np.linspace(np.min(radiuses), np.max(radiuses), 1000)
theoretical_density = density(x, a, N)

# Построение графиков
plt.plot(x, theoretical_density, label='Theoretical Density', color='blue')
plt.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), alpha=0.5, label='Empirical Density', color='green')

plt.xlabel('Radius')
plt.ylabel('Density')
plt.title('Density Plot')
plt.legend()
plt.show()
