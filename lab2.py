import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fields = 300
x1 = np.linspace(-100, 100, fields)
x2 = np.linspace(-20, 50, fields)
y = x2 / (1 + np.exp(-x1))

df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y,
})

df.to_csv('num_data.csv')

data = pd.read_csv('num_data.csv')

for column in data.columns:
    print(f" среднее: {data[column].mean():.3f}")
    print(f" максимальное: {data[column].max():.3f}")
    print(f" минимальное: {data[column].min():.3f}")
    print()

plt.figure(figsize=(12, 6))

#график 1
plt.subplot(1, 2, 1)
x2_const = data['x2'].mean()
y_vs_x1 = x2_const / (1 + np.exp(-data['x1']))
plt.scatter(data['x1'], y_vs_x1, s=10)
plt.xlabel('x1')
plt.ylabel('y')
plt.grid(True)

# график 2
plt.subplot(1, 2, 2)
x1_const = data['x1'].mean()
y_vs_x2 = data['x2'] / (1 + np.exp(-x1_const))
plt.scatter(data['x2'], y_vs_x2, s=10)
plt.xlabel('x2')
plt.ylabel('y')
plt.grid(True)

plt.show()


#фильтрация по условию
mean_x1 = data['x1'].mean()
mean_x2 = data['x2'].mean()
filtered_data = data[(data['x1'] < mean_x1) | (data['x2'] < mean_x2)]

filtered_data.to_csv('filtered_data.csv')
print(f"изначально было строк: {len(data)}, стало строк: {len(filtered_data)}")


#  3d плоскость
plt.figure(figsize=(12, 6))

# сетка
X1, X2 = np.meshgrid(np.linspace(-100, 100, 50), 
                     np.linspace(-20, 50, 50))
Y = X2 / (1 + np.exp(-X1))

# контурный график
contour = plt.contourf(X1, X2, Y, levels=20, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='y')

plt.contour(X1, X2, Y, levels=10, colors='black', alpha=0.3, linewidths=0.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Плоскость y = x2 / (1 + exp(-x1))')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
