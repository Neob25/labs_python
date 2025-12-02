import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Размерность данных: {data.shape}")
        print("\nИнформация о данных:")
        print(data.info())
        return data
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

# Предварительная обработка данных
def preprocess_data(data):
    print("Пропущенные значения:")
    print(data.isnull().sum())

    data_clean = data.dropna()
    print(f"\nДанные после удаления пропущенных значений: {data_clean.shape}")
    
    numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
    print(f"\nЧисловые колонки: {list(numeric_columns)}")
    
    # Если числовых колонок мало
    if len(numeric_columns) < 3:
        for i in range(5):
            data_clean[f'feature_{i}'] = np.random.normal(0, 1, len(data_clean))
        
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
        print(f"Новые числовые колонки: {list(numeric_columns)}")
    
    # Выбор данных для кластеризации
    X = data_clean[numeric_columns]
    
    print(f"\nФинальная размерность данных для кластеризации: {X.shape}")
    return X, data_clean

# Масштабирование
def scale_features(X):
    
    # StandardScaler
    scaler_standard = StandardScaler()
    X_standard = scaler_standard.fit_transform(X)
    print("StandardScaler применен успешно")
    
    # MinMaxScaler
    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X)
    print("онрмализация применена успешно")
    
    return X_standard, X_minmax

# Визуализация данных до и после масштабирования
def visualize_scaling(X, X_standard, X_minmax, feature_names):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    pd.DataFrame(X, columns=feature_names).boxplot(ax=axes[0])
    axes[0].set_title('Исходные данные ')
    axes[0].tick_params(axis='x', rotation=45)
    
    pd.DataFrame(X_standard, columns=feature_names).boxplot(ax=axes[1])
    axes[1].set_title('После StandardScaler')
    axes[1].tick_params(axis='x', rotation=45)
    
    pd.DataFrame(X_minmax, columns=feature_names).boxplot(ax=axes[2])
    axes[2].set_title('После нормализации')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Кластеризация с использованием K-means
def kmeans_clustering(X, max_clusters=10):
    
    # Метод локтя
    wcss = []
    silhouette_scores = []
    
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Визуализация метода и silhouette score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Метод локтя
    ax1.plot(range(2, max_clusters + 1), wcss, marker='o')
    ax1.set_title('Метод локтя для K-means')
    ax1.set_xlabel('Количество кластеров')
    ax1.set_ylabel('WCSS')
    
    # Silhouette score
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='red')
    ax2.set_title('Silhouette Score для K-means')
    ax2.set_xlabel('Количество кластеров')
    ax2.set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.show()
    
    # Выбор оптимального числа кластеров на основе silhouette score
    optimal_clusters = np.argmax(silhouette_scores) + 2  # начинаем с 2 кластеров
    print(f"Оптимальное число кластеров (по silhouette score): {optimal_clusters}")

    kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans_optimal.fit_predict(X)
    
    # Оценка качества кластеризации
    silhouette = silhouette_score(X, kmeans_labels)
    calinski = calinski_harabasz_score(X, kmeans_labels)
    davies = davies_bouldin_score(X, kmeans_labels)
    
    print(f"Результаты K-means:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"  - Calinski-Harabasz Score: {calinski:.4f}")
    print(f"  - Davies-Bouldin Score: {davies:.4f}")
    
    return kmeans_labels, optimal_clusters, silhouette, calinski, davies

# Кластеризация с DBSCAN
def dbscan_clustering(X):
    
    best_silhouette = -1
    best_eps = 0.1
    best_min_samples = 5
    best_labels = None
    
    for eps in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for min_samples in [3, 5, 7, 10]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Пропускаем случаи, когда все точки в одном кластере или шуме
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(X) // 2:
                silhouette = silhouette_score(X, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
    
    if best_labels is None:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        best_labels = dbscan.fit_predict(X)
        best_eps = 0.5
        best_min_samples = 5
    
    # Оценка качества кластеризации
    unique_labels = np.unique(best_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(best_labels).count(-1)
    
    print(f"Оптимальные параметры DBSCAN: eps={best_eps}, min_samples={best_min_samples}")
    print(f"Количество кластеров: {n_clusters}")
    print(f"Количество шумовых точек: {n_noise}")
    
    if n_clusters > 1:
        silhouette = silhouette_score(X, best_labels)
        calinski = calinski_harabasz_score(X, best_labels)
        davies = davies_bouldin_score(X, best_labels)
    else:
        silhouette = -1
        calinski = -1
        davies = float('inf')
    
    print(f"Результаты DBSCAN:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"  - Calinski-Harabasz Score: {calinski:.4f}")
    print(f"  - Davies-Bouldin Score: {davies:.4f}")
    
    return best_labels, n_clusters, silhouette, calinski, davies

# Кластеризация с Agglomerative Clustering
def agglomerative_clustering(X, max_clusters=10):
    
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglo.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Визуализация silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score для Agglomerative Clustering')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    optimal_clusters = np.argmax(silhouette_scores) + 2
    print(f"Оптимальное число кластеров: {optimal_clusters}")

    agglo_optimal = AgglomerativeClustering(n_clusters=optimal_clusters)
    agglo_labels = agglo_optimal.fit_predict(X)
    
    # Оценка качества кластеризации
    silhouette = silhouette_score(X, agglo_labels)
    calinski = calinski_harabasz_score(X, agglo_labels)
    davies = davies_bouldin_score(X, agglo_labels)
    
    print(f"Результаты Agglomerative Clustering:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"  - Calinski-Harabasz Score: {calinski:.4f}")
    print(f"  - Davies-Bouldin Score: {davies:.4f}")
    
    return agglo_labels, optimal_clusters, silhouette, calinski, davies

# Визуализация кластеризации
def visualize_clustering_results(X, results, algorithm_names):
    
    # Применение PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Применение t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    fig, axes = plt.subplots(2, len(algorithm_names), figsize=(15, 10))
    
    if len(algorithm_names) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (algo_name, labels) in enumerate(zip(algorithm_names, results)):
        # Визуализация с PCA
        scatter_pca = axes[0, i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[0, i].set_title(f'{algo_name} (PCA)')
        axes[0, i].set_xlabel('PC1')
        axes[0, i].set_ylabel('PC2')
        plt.colorbar(scatter_pca, ax=axes[0, i])
        
        # Визуализация с t-SNE
        scatter_tsne = axes[1, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[1, i].set_title(f'{algo_name} (t-SNE)')
        axes[1, i].set_xlabel('t-SNE1')
        axes[1, i].set_ylabel('t-SNE2')
        plt.colorbar(scatter_tsne, ax=axes[1, i])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Объясненная дисперсия PCA: {pca.explained_variance_ratio_.sum():.4f}")

# Сравнение алгоритмов
def compare_algorithms(results_df):
    
    print("\nСравнительная таблица алгоритмов:")
    print(results_df.to_string(index=False))
    
    metrics = ['Silhouette', 'Calinski_Harabasz', 'Davies_Bouldin']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        if metric != 'Davies_Bouldin':  # Для Davies-Bouldin меньше = лучше
            axes[i].bar(results_df['Algorithm'], results_df[metric])
            axes[i].set_title(f'{metric} Score (больше = лучше)')
        else:
            axes[i].bar(results_df['Algorithm'], results_df[metric])
            axes[i].set_title(f'{metric} Score (меньше = лучше)')
        
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Определение лучшего алгоритма
    best_silhouette_idx = results_df['Silhouette'].idxmax()
    best_calinski_idx = results_df['Calinski_Harabasz'].idxmax()
    best_davies_idx = results_df['Davies_Bouldin'].idxmin()
    
    print(f"\nЛУЧШИЕ АЛГОРИТМЫ:")
    print(f"По Silhouette Score: {results_df.loc[best_silhouette_idx, 'Algorithm']} ({results_df.loc[best_silhouette_idx, 'Silhouette']:.4f})")
    print(f"По Calinski-Harabasz Score: {results_df.loc[best_calinski_idx, 'Algorithm']} ({results_df.loc[best_calinski_idx, 'Calinski_Harabasz']:.4f})")
    print(f"По Davies-Bouldin Score: {results_df.loc[best_davies_idx, 'Algorithm']} ({results_df.loc[best_davies_idx, 'Davies_Bouldin']:.4f})")


def main():
    file_path = r"C:\Users\Артём\Desktop\уник\MACHINE\aaai+2013+accepted+papers\[UCI] AAAI-13 Accepted Papers - Papers.csv"
    
    print("ПРОГРАММА КЛАСТЕРИЗАЦИИ ДАННЫХ AAAI-2013")
    print("="*70)
    print(f"Путь к данным: {file_path}")
    
    data = load_data(file_path)
    if data is None:
        print("Не удалось загрузить данные. Создаем демонстрационные данные...")
        # Создание демонстрационных данных
        np.random.seed(42)
        n_samples = 200
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples),
            'feature5': np.random.normal(0, 1, n_samples)
        })
        print("Демонстрационные данные созданы")
    
    X, data_clean = preprocess_data(data)

    X_standard, X_minmax = scale_features(X)
    
    # Визуализация масштабирования
    if X.shape[1] > 5:
        feature_names = X.columns[:5] if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(5)]
        visualize_scaling(X.values[:, :5], X_standard[:, :5], X_minmax[:, :5], feature_names)
    else:
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
        visualize_scaling(X.values, X_standard, X_minmax, feature_names)
    
    X_for_clustering = X_standard
    
    # Применение алгоритмов кластеризации
    all_results = []
    algorithm_names = []
    metrics_data = []
    
    #K-means
    print("\n" + "="*70)
    print("ЗАПУСК АЛГОРИТМА K-MEANS")
    print("="*70)
    kmeans_labels, kmeans_clusters, kmeans_silhouette, kmeans_calinski, kmeans_davies = kmeans_clustering(X_for_clustering)
    all_results.append(kmeans_labels)
    algorithm_names.append('K-Means')
    metrics_data.append({
        'Algorithm': 'K-Means',
        'Clusters': kmeans_clusters,
        'Silhouette': kmeans_silhouette,
        'Calinski_Harabasz': kmeans_calinski,
        'Davies_Bouldin': kmeans_davies
    })
    
    #DBSCAN
    print("\n" + "="*70)
    print("ЗАПУСК АЛГОРИТМА DBSCAN")
    print("="*70)
    dbscan_labels, dbscan_clusters, dbscan_silhouette, dbscan_calinski, dbscan_davies = dbscan_clustering(X_for_clustering)
    all_results.append(dbscan_labels)
    algorithm_names.append('DBSCAN')
    metrics_data.append({
        'Algorithm': 'DBSCAN',
        'Clusters': dbscan_clusters,
        'Silhouette': dbscan_silhouette,
        'Calinski_Harabasz': dbscan_calinski,
        'Davies_Bouldin': dbscan_davies
    })
    
    #Agglomerative Clustering
    print("\n" + "="*70)
    print("ЗАПУСК АЛГОРИТМА AGGLOMERATIVE CLUSTERING")
    print("="*70)
    agglo_labels, agglo_clusters, agglo_silhouette, agglo_calinski, agglo_davies = agglomerative_clustering(X_for_clustering)
    all_results.append(agglo_labels)
    algorithm_names.append('Agglomerative')
    metrics_data.append({
        'Algorithm': 'Agglomerative',
        'Clusters': agglo_clusters,
        'Silhouette': agglo_silhouette,
        'Calinski_Harabasz': agglo_calinski,
        'Davies_Bouldin': agglo_davies
    })
    
    #DataFrame с результатами
    results_df = pd.DataFrame(metrics_data)   
    visualize_clustering_results(X_for_clustering, all_results, algorithm_names)
    
    # Сравнение алгоритмов
    compare_algorithms(results_df)
    
    try:
        #метки кластеров к исходным данным
        for i, (algo_name, labels) in enumerate(zip(algorithm_names, all_results)):
            data_clean[f'cluster_{algo_name.lower()}'] = labels
        
        output_path = r"C:\Users\Артём\Desktop\уник\MACHINE\clustering_results.csv"
        data_clean.to_csv(output_path, index=False)
    except Exception as e:
        print(f"\nНе удалось сохранить результаты: {e}")

if __name__ == "__main__":
    main()
