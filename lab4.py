import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train_data = pd.read_csv(r'C:\Users\Артём\Desktop\уник\MACHINE\pen+based+recognition+of+handwritten+digits\pendigits.tra', header=None)
    test_data = pd.read_csv(r'C:\Users\Артём\Desktop\уник\MACHINE\pen+based+recognition+of+handwritten+digits\pendigits.tes', header=None)
    
    return train_data, test_data

# Разделение на признаки и целевую переменную
def prepare_data(train_data, test_data):
    x_train_full = train_data.iloc[:, :-1].values
    y_train_full = train_data.iloc[:, -1].values
    
    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    return x_train_full, y_train_full, x_test, y_test

# Разделение на обучающую и валидационную 
def split_validation_set(x_train_full, y_train_full, test_size=0.2, random_state=42):
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_train_full
    )
    return x_train, x_val, y_train, y_val

# Масштабирование
def scale_features(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler

# Perceptron
def train_perceptron(x_train, y_train, x_val, y_val, alpha=0.0001, penalty='l2', eta0=0.1, max_iter=1000):
    perceptron = Perceptron(
         alpha=alpha,       # Сила регуляризации 
        penalty=penalty,    # Тип регуляризации (L1, L2)
        eta0=eta0,          # Начальная скорость обучения
        max_iter=max_iter,  # Максимальное количество итераций обучения
        random_state=42     # Для воспроизводимости
    )
    
    perceptron.fit(x_train, y_train)
    
    y_val_pred = perceptron.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    return perceptron, val_accuracy

# MLPClassifier
def train_mlp(x_train, y_train, x_val, y_val, hidden_layer_sizes=(100,), activation='relu', 
              solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=1000):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42
    )
    
    mlp.fit(x_train, y_train)
    
    y_val_pred = mlp.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    return mlp, val_accuracy


def run_experiments(x_train_scaled, y_train, x_val_scaled, y_val):
    print("=" * 60)
    print("ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ")
    
    learning_rates = [0.001, 0.01, 0.1]
    alphas = [0.0001, 0.001, 0.01]
    solvers = ['adam', 'sgd']
    
    best_perceptron = None
    best_perceptron_score = 0
    best_perceptron_params = {}
    
    best_mlp = None
    best_mlp_score = 0
    best_mlp_params = {}
    
    # Эксперименты с Perceptron
    print("\nPERCEPTRON ЭКСПЕРИМЕНТЫ:")
    print("-" * 40)
    
    for eta0 in learning_rates:
        for alpha in alphas:
            for penalty in ['l2', 'l1', 'elasticnet']:
                try:
                    perceptron, score = train_perceptron(
                        x_train_scaled, y_train, x_val_scaled, y_val,
                        alpha=alpha, penalty=penalty, eta0=eta0
                    )
                    
                    print(f"eta0: {eta0}, alpha: {alpha}, penalty: {penalty} -> Accuracy: {score:.4f}")
                    
                    if score > best_perceptron_score:
                        best_perceptron_score = score
                        best_perceptron = perceptron
                        best_perceptron_params = {
                            'eta0': eta0,
                            'alpha': alpha,
                            'penalty': penalty
                        }
                except:
                    continue
    
    # Эксперименты с MLPClassifier
    print("\nMLPClassifier ЭКСПЕРИМЕНТЫ:")
    print("-" * 40)
    
    for learning_rate in learning_rates:
        for alpha in alphas:
            for solver in solvers:
                for activation in ['relu', 'tanh']:
                    mlp, score = train_mlp(
                        x_train_scaled, y_train, x_val_scaled, y_val,
                        learning_rate_init=learning_rate,
                        alpha=alpha,
                        solver=solver,
                        activation=activation
                    )
                    
                    print(f"learning_rate: {learning_rate}, alpha: {alpha}, solver: {solver}, activation: {activation} -> Accuracy: {score:.4f}")
                    
                    if score > best_mlp_score:
                        best_mlp_score = score
                        best_mlp = mlp
                        best_mlp_params = {
                            'learning_rate_init': learning_rate,
                            'alpha': alpha,
                            'solver': solver,
                            'activation': activation
                        }
    
    return best_perceptron, best_perceptron_params, best_perceptron_score, best_mlp, best_mlp_params, best_mlp_score

def main():
    train_data, test_data = load_data()
    
    print(f"Размер обучающей выборки: {train_data.shape}")
    print(f"Размер тестовой выборки: {test_data.shape}")
    
    x_train_full, y_train_full, x_test, y_test = prepare_data(train_data, test_data)
    x_train, x_val, y_train, y_val = split_validation_set(x_train_full, y_train_full)
    
    print(f"Обучающая выборка: {x_train.shape[0]} samples")
    print(f"Валидационная выборка: {x_val.shape[0]} samples")
    print(f"Тестовая выборка: {x_test.shape[0]} samples")
    
    # Масштабирование признаков
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_features(x_train, x_val, x_test)
    print("\nМасштабирование")
    
    best_perceptron, best_perceptron_params, best_perceptron_score, best_mlp, best_mlp_params, best_mlp_score = run_experiments(x_train_scaled, y_train, x_val_scaled, y_val)
    
    print("\n" + "=" * 60)
    print("ЛУЧШИЕ ПАРАМЕТРЫ")
    print(f"Perceptron - Лучшая точность на валидации: {best_perceptron_score:.4f}")
    print(f"Параметры: {best_perceptron_params}")
    
    print(f"\nMLPClassifier - Лучшая точность на валидации: {best_mlp_score:.4f}")
    print(f"Параметры: {best_mlp_params}")
    
    print("\n" + "=" * 60)
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    
    y_test_pred_perceptron = best_perceptron.predict(x_test_scaled)
    test_accuracy_perceptron = accuracy_score(y_test, y_test_pred_perceptron)
    print(f"Perceptron точность на тестовой выборке: {test_accuracy_perceptron:.4f}")

    y_test_pred_mlp = best_mlp.predict(x_test_scaled)
    test_accuracy_mlp = accuracy_score(y_test, y_test_pred_mlp)
    print(f"MLPClassifier точность на тестовой выборке: {test_accuracy_mlp:.4f}")
    
    print("\nPerceptron:")
    print(classification_report(y_test, y_test_pred_perceptron))
    
    print("\nMLPClassifier:")
    print(classification_report(y_test, y_test_pred_mlp))

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ")
    if test_accuracy_perceptron > test_accuracy_mlp:
        print(f"Perceptron показал лучшую производительность: {test_accuracy_perceptron:.4f}")
    elif test_accuracy_mlp > test_accuracy_perceptron:
        print(f"MLPClassifier показал лучшую производительность: {test_accuracy_mlp:.4f}")
    else:
        print("Обе модели показали одинаковую производительность")

if __name__ == "__main__":
    main()
