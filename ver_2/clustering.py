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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def perform_clustering(X, max_k=10, use_pca=True):
    """Кластеризация MiniBatchKMeans с автоподбором k по силуэту."""
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
    """Эллипсы вокруг кластеров по PCA-проекции."""
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


def visualize_clusters(X, labels, feature_names, output_dir, model=None,
                       max_scatter_points=10000, max_silhouette_points=20000):
    """
    Строит и сохраняет три графика с ускорением через подвыборки.

    max_scatter_points – макс. число точек на PCA scatter (остальное семплируется)
    max_silhouette_points – макс. число точек для силуэта
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))
    n_total = X.shape[0]

    # ---------- 1. PCA scatter с эллипсами (подвыборка) ----------
    print("Построение PCA графика (может занять несколько секунд)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca_all = pca.fit_transform(X)

    # Случайная подвыборка для отрисовки точек
    if n_total > max_scatter_points:
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(n_total, size=max_scatter_points, replace=False)
        X_pca_sample = X_pca_all[idx_sample]
        labels_sample = labels[idx_sample]
    else:
        X_pca_sample = X_pca_all
        labels_sample = labels

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Рисуем только подвыборку
    for i, label in enumerate(unique_labels):
        mask = labels_sample == label
        ax.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1],
                   color=palette[i], label=f'Кластер {label}',
                   alpha=0.6, edgecolor='k', s=20)

    # Эллипсы и центроиды считаем по полным данным (быстро)
    plot_cluster_ellipses(ax, X_pca_all, labels, palette, n_std=1.5, alpha=0.15)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        center = np.mean(X_pca_all[mask], axis=0)
        ax.scatter(*center, color=palette[i], edgecolor='black', s=200, marker='D',
                   linewidth=1.5, label=f'Центроид {label}')

    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Первая главная компонента')
    ax.set_ylabel('Вторая главная компонента')
    ax.set_title('Проекция данных на первые две главные компоненты (PCA)\nс эллипсами кластеров (подвыборка)')
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_pca.png', dpi=150)
    plt.close()
    print("PCA график сохранён.")

    # ---------- 2. Silhouette plot (подвыборка) ----------
    print("Построение графика силуэта (может занять несколько секунд)...")
    # Вычисляем silhouette_samples для всех? Медленно. Лучше на подвыборке.
    if n_total > max_silhouette_points:
        rng = np.random.RandomState(42)
        idx_sil = rng.choice(n_total, size=max_silhouette_points, replace=False)
        X_sub = X[idx_sil]
        labels_sub = labels[idx_sil]
    else:
        X_sub = X
        labels_sub = labels

    # Быстрое вычисление
    sample_sil_values = silhouette_samples(X_sub, labels_sub)

    plt.figure(figsize=(10, 7))
    y_lower = 10
    for i, label in enumerate(unique_labels):
        cluster_vals = sample_sil_values[labels_sub == label]
        cluster_vals.sort()
        size_cluster_i = cluster_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_vals,
                          facecolor=palette[i], alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10

    avg_sil = silhouette_score(X_sub, labels_sub, sample_size=min(10000, X_sub.shape[0]), n_jobs=-1, random_state=42)
    plt.axvline(x=avg_sil, color="red", linestyle="--")
    plt.xlabel('Коэффициент силуэта')
    plt.ylabel('Кластер')
    plt.title('График силуэта для кластеров (подвыборка)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_silhouette.png', dpi=150)
    plt.close()
    print("График силуэта сохранён.")

    # ---------- 3. Средние значения признаков (топ-10) ----------
    print("Построение графика средних значений признаков...")
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
        print("График средних значений сохранён.")


def profile_clusters(snapshot_with_labels):
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
    counts = snapshot_with_labels['cluster'].value_counts().sort_index()
    total = len(snapshot_with_labels)
    summary = pd.DataFrame({
        'cluster': counts.index,
        'count': counts.values,
        'percentage': (counts.values / total * 100).round(2)
    })
    return summary

def feature_importance_analysis(X, labels, feature_names):
    """
    Обучает RandomForestClassifier предсказывать метку кластера,
    возвращает DataFrame с важностью признаков, отсортированный по убыванию.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, labels)
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return importance_df


def cluster_portraits(snapshot, feature_names, top_n=5):
    """
    Формирует словесные портреты кластеров на основе ключевых признаков.
    """
    if 'cluster' not in snapshot.columns:
        return {}

    # Выбираем признаки, которые есть в snapshot и не являются бинарными контактами (для них свой подход)
    numeric_cols = [f for f in feature_names if f in snapshot.columns]
    if not numeric_cols:
        return {}

    # Общие средние по всем данным
    overall_mean = snapshot[numeric_cols].mean()
    cluster_means = snapshot.groupby('cluster')[numeric_cols].mean()

    # Важность признаков через RandomForest
    X = snapshot[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, snapshot['cluster'])
    importances = clf.feature_importances_
    importance_series = pd.Series(importances, index=numeric_cols).sort_values(ascending=False)
    top_features = importance_series.head(top_n).index.tolist()

    portraits = {}
    for cluster in sorted(snapshot['cluster'].unique()):
        desc_parts = []
        for col in top_features:
            mean_cl = cluster_means.loc[cluster, col]
            mean_all = overall_mean[col]
            # Для бинарных (0/1) признаков выводим процент
            if snapshot[col].nunique() <= 2 and snapshot[col].min() >= 0 and snapshot[col].max() <= 1:
                pct_cl = mean_cl * 100
                pct_all = mean_all * 100
                diff = pct_cl - pct_all
                if abs(diff) < 5:
                    desc = f"{col}: {pct_cl:.0f}% (как в среднем)"
                else:
                    direction = "больше" if diff > 0 else "меньше"
                    desc = f"{col}: {pct_cl:.0f}% ({direction} на {abs(diff):.0f} п.п.)"
                desc_parts.append(desc)
                continue

            # Для количественных признаков: сравниваем относительное отклонение
            if abs(mean_all) > 1e-6:
                relative_diff = (mean_cl - mean_all) / abs(mean_all)
                if abs(relative_diff) < 0.2:
                    desc = f"{col}: {mean_cl:.1f} (на уровне среднего)"
                elif relative_diff > 2.0:
                    desc = f"{col}: {mean_cl:.1f} (в {relative_diff+1:.1f} раза выше среднего)"
                elif relative_diff < -0.5:
                    desc = f"{col}: {mean_cl:.1f} (в {1/(1+relative_diff):.1f} раза ниже среднего)"
                else:
                    direction = "выше" if relative_diff > 0 else "ниже"
                    desc = f"{col}: {mean_cl:.1f} ({direction} среднего на {abs(relative_diff)*100:.0f}%)"
            else:
                desc = f"{col}: {mean_cl:.1f}"
            desc_parts.append(desc)

        # Добавляем общие сводки
        if 'debt_amount' in cluster_means.columns:
            desc_parts.append(f"Средний долг: {cluster_means.loc[cluster, 'debt_amount']:.0f} руб.")
        if 'debt_age' in cluster_means.columns:
            desc_parts.append(f"возраст долга: {cluster_means.loc[cluster, 'debt_age']:.1f} мес.")
        if 'num_contacts' in cluster_means.columns:
            desc_parts.append(f"контактов: {cluster_means.loc[cluster, 'num_contacts']:.1f}")

        portraits[cluster] = "Кластер {}: ".format(cluster) + "; ".join(desc_parts)

    return portraits