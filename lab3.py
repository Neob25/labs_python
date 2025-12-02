import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

path1 = r"C:\Users\Артём\Desktop\уник\MACHINE\sml2010\NEW-DATA-1.T15.txt"
path2 = r"C:\Users\Артём\Desktop\уник\MACHINE\sml2010\NEW-DATA-2.T15.txt"

df1 = pd.read_csv(path1, sep=r"\s+", comment="#", header=None)
df2 = pd.read_csv(path2, sep=r"\s+", comment="#", header=None)
df = pd.concat([df1, df2], ignore_index=True)

cols = [
    "Date", "Time", "Temp_Comedor", "Temp_Habitacion", "Weather_Temp",
    "CO2_Comedor", "CO2_Habitacion", "Hum_Comedor", "Hum_Habitacion",
    "Light_Comedor", "Light_Habitacion", "Precipitacion", "Crepusculo",
    "Viento", "Sol_Oest", "Sol_Est", "Sol_Sud", "Piranometro",
    "Entalpic_1", "Entalpic_2", "Entalpic_turbo", "Temp_Exterior",
    "Hum_Exterior", "Day_Of_Week"
]
df.columns = cols
df = df.dropna()

X = df[["Temp_Comedor"]].values
y = df["Temp_Habitacion"].values

test_size = int(0.2 * len(X))  
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

#Линейная регрессия
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("линейная")
print(f"Среднеквадратичная ошибка: {mse:.4f}")
print(f"Коэффициент детерминации: {r2:.4f}")

#График линейной регрессии 
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', label="Тестовые данные")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Предсказания")
plt.xlabel("Температура в comedor")
plt.ylabel("Температура в комнате")
plt.title("тестовая выборка")
plt.legend()
plt.grid()
plt.show()

#Полиномиальная регрессия
degrees = range(1, 8)
train_scores, test_scores = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    reg = LinearRegression()
    reg.fit(X_poly_train, y_train)

    train_scores.append(r2_score(y_train, reg.predict(X_poly_train)))
    test_scores.append(r2_score(y_test, reg.predict(X_poly_test)))

plt.figure(figsize=(8,5))
plt.plot(degrees, train_scores, marker='o', label="Train R^2")
plt.plot(degrees, test_scores, marker='o', label="Test R^2")
plt.xlabel("Степень полинома")
plt.ylabel("R^2")
plt.title("Полиномиальная регрессия")
plt.legend()
plt.grid()
plt.show()

#Регуляризация
alphas = np.logspace(-3, 3, 20)
train_r2, test_r2 = [], []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    train_r2.append(r2_score(y_train, ridge.predict(X_train)))
    test_r2.append(r2_score(y_test, ridge.predict(X_test)))

plt.figure(figsize=(8,5))
plt.semilogx(alphas, train_r2, marker='o', label="Train R^2")
plt.semilogx(alphas, test_r2, marker='o', label="Test R^2")
plt.xlabel("Коэффициент alpha")
plt.ylabel("R^2")
plt.title("Ridge-регрессия")
plt.legend()
plt.grid()
plt.show()
