import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
from pathlib import Path

def perform_clustering(X, max_k=10, use_pca=True):
    if use_pca and X.shape[1] > 15:
        pca = PCA(n_components=0.95, random_state=42)
        X = pca.fit_transform(X)
        print(f"PCA reduced features to {X.shape[1]}")

    best_k = 4
    best_score = -1.0
    best_model = None
    best_labels = None

    for k in range(4, max_k + 1):
        km = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=2048,
            n_init=3, max_iter=20, reassignment_ratio=0.01
        )
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels, sample_size=10000, n_jobs=-1, random_state=42)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = km
                best_labels = labels

    if best_model is None:
        best_model = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=2048, n_init=3)
        best_labels = best_model.fit_predict(X)

    db_index = davies_bouldin_score(X, best_labels) if len(set(best_labels)) > 1 else np.nan

    print(f"Лучшее число кластеров: {best_k}")
    print(f"Silhouette Score: {best_score:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")

    return best_model, best_labels, {'silhouette': best_score, 'davies_bouldin': db_index}


def plot_cluster_ellipses(ax, X_pca, labels, palette, n_std=1.5, alpha=0.15):
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        cluster_points = X_pca[labels == label]
        if len(cluster_points) > 1:
            cov = np.cov(cluster_points, rowvar=False)
            if cov.shape == (2, 2) and np.linalg.det(cov) > 1e-6:
                center = np.mean(cluster_points, axis=0)
                v, w = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
                width, height = 2 * n_std * np.sqrt(v)
                ellipse = Ellipse(xy=center, width=width, height=height,
                                  angle=angle, edgecolor=palette[i], fc=palette[i],
                                  lw=2, alpha=alpha)
                ax.add_patch(ellipse)


def visualize_clusters(X, labels, feature_names, output_dir, model=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))

    # 1. PCA scatter с эллипсами
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette[i], label=f'Кластер {label}',
                   alpha=0.6, edgecolor='k', s=20)
    plot_cluster_ellipses(ax, X_pca, labels, palette, n_std=1.5, alpha=0.15)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        center = np.mean(X_pca[mask], axis=0)
        ax.scatter(*center, color=palette[i], edgecolor='black', s=200, marker='D',
                   linewidth=1.5, label=f'Центроид {label}')
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Первая главная компонента')
    ax.set_ylabel('Вторая главная компонента')
    ax.set_title('Проекция данных на первые две главные компоненты (PCA)\nс эллипсами кластеров')
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_pca.png', dpi=150)
    plt.close()

    # 2. Silhouette plot
    sample_silhouette_values = silhouette_samples(X, labels)
    plt.figure(figsize=(10, 7))
    y_lower = 10
    for i, label in enumerate(unique_labels):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == label]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = palette[i]
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10
    avg_sil = silhouette_score(X, labels)
    plt.axvline(x=avg_sil, color="red", linestyle="--")
    plt.xlabel('Коэффициент силуэта')
    plt.ylabel('Кластер')
    plt.title('График силуэта для кластеров')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_silhouette.png', dpi=150)
    plt.close()

    # 3. Средние значения признаков (топ-10)
    if feature_names is not None and len(feature_names) > 1:
        cluster_means = []
        for label in unique_labels:
            cluster_means.append(np.mean(X[labels == label], axis=0))
        cluster_means = np.array(cluster_means)
        variance_per_feature = np.var(cluster_means, axis=0)
        top_indices = np.argsort(variance_per_feature)[-10:]
        top_features = [feature_names[i] for i in top_indices]
        top_means = cluster_means[:, top_indices]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(top_features))
        width = 0.8 / len(unique_labels)
        for i, label in enumerate(unique_labels):
            plt.bar(x + i * width, top_means[i], width, label=f'Кластер {label}', color=palette[i])
        plt.xticks(x + width * (len(unique_labels) - 1) / 2, top_features, rotation=45, ha='right')
        plt.ylabel('Среднее значение (стандартизованное)')
        plt.title('Средние значения наиболее вариативных признаков по кластерам')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'clusters_feature_means.png', dpi=150)
        plt.close()


def profile_clusters(snapshot_with_labels):
    """Средние характеристики кластеров."""
    return snapshot_with_labels.groupby('cluster').agg({
        'debt_amount': 'mean',
        'debt_age': 'mean',
        'avg_payment_3m': 'mean',
        'num_contacts': 'mean',
        'has_phone': 'mean',
        'has_email': 'mean',
        'prev_restriction_notice': 'mean'
    }).round(2)


def cluster_summary(snapshot_with_labels):
    """Количество и процент клиентов в каждом кластере."""
    counts = snapshot_with_labels['cluster'].value_counts().sort_index()
    total = len(snapshot_with_labels)
    summary = pd.DataFrame({
        'cluster': counts.index,
        'count': counts.values,
        'percentage': (counts.values / total * 100).round(2)
    })
    return summary