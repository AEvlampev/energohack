import pandas as pd
import argparse
import json
from pathlib import Path
import config as cfg
from preprocessing import load_all_data, build_monthly_snapshot
from feature_engineering import prepare_clustering_features, add_cluster_labels
from clustering import (
    perform_clustering, visualize_clusters, profile_clusters, cluster_summary, cluster_portraits
)
from model import train_regression_model, train_classification_model
from optimizer import greedy_optimize
from explainer import generate_explanations
import warnings
warnings.filterwarnings('ignore')

def main(target_date_str):
    target_month = pd.to_datetime(target_date_str)
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print("Загрузка данных...")
    data = load_all_data()

    print("Формирование снимка на", target_date_str)
    snapshot = build_monthly_snapshot(data, target_month)
    snapshot = snapshot[snapshot['debt_amount'] > 0].copy()
    print(f"Найдено должников: {len(snapshot)}")

    # ---------- Кластеризация ----------
    print("Подготовка признаков для кластеризации...")
    X, feat_names, preprocessor = prepare_clustering_features(snapshot)

    print("Кластеризация...")
    if len(snapshot) < 4:
        snapshot['cluster'] = 0
        clust_metrics = {'silhouette': None, 'davies_bouldin': None}
    else:
        model, labels, clust_metrics = perform_clustering(X, max_k=10)
        snapshot = add_cluster_labels(snapshot, labels)
        print("Создание графиков кластеризации...")
        visualize_clusters(X, labels, feat_names, output_dir, model=model)

    # Статистика по кластерам
    clust_summary = cluster_summary(snapshot)
    print("\nРаспределение клиентов по кластерам:")
    print(clust_summary.to_string(index=False))

    # После cluster_summary
    print("\nПортреты кластеров (ключевые отличия):")
    # Для портретов используем исходные признаки, которые использовались в кластеризации,
    # но также можно добавить признаки из модели, например debt_amount.
    # Возьмём признаковый набор из кластеризации
    portrait_features = feat_names  # feat_names получены из prepare_clustering_features
    portraits = cluster_portraits(snapshot, portrait_features, top_n=5)
    for cl, desc in portraits.items():
        print(desc)

    print("\nПрофили кластеров (средние):")
    profiles = profile_clusters(snapshot)
    print(profiles)

    with open(output_dir / 'metrics_clustering.json', 'w') as f:
        json.dump(clust_metrics, f, indent=2)

    # ---------- Обучение моделей ----------
    feature_cols = [
        'debt_amount', 'debt_age', 'num_contacts',
        'has_phone', 'has_email', 'has_benefits', 'gasification',
        'mkd', 'dormitory', 'trend_debt',
        'prev_restriction_notice', 'prev_restriction', 'prev_court_order',
        'has_info_measures', 'has_restriction_measures'
    ]
    available_feats = [f for f in feature_cols if f in snapshot.columns]

    if 'recovery_rate' not in snapshot.columns:
        snapshot['recovery_rate'] = 0.0
    if 'paid_next_month' not in snapshot.columns:
        snapshot['paid_next_month'] = (snapshot.get('last_payment_amount', 0) > 0).astype(int)

    print("\n=== Обучение регрессионной модели (recovery_rate) ===")
    reg_model, reg_metrics = train_regression_model(snapshot, available_feats, 'recovery_rate')
    print("Метрики регрессии:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")
    with open(output_dir / 'metrics_regression.json', 'w') as f:
        json.dump(reg_metrics, f, indent=2)

    print("\n=== Обучение классификационной модели (paid_next_month) ===")
    clf_model, clf_metrics = train_classification_model(snapshot, available_feats, 'paid_next_month')
    print("Метрики классификации:")
    for k, v in clf_metrics.items():
        print(f"  {k}: {v:.4f}")
    with open(output_dir / 'metrics_classification.json', 'w') as f:
        json.dump(clf_metrics, f, indent=2)

    # ---------- Оптимизация ----------
    print("\nОптимизация назначений...")
    effectiveness = {}
    recommendations, rec_df = greedy_optimize(
        snapshot, effectiveness, cfg.MONTHLY_LIMITS, cfg.MEASURE_CRITERIA,
        model=reg_model, feature_cols=available_feats
    )

    # ---------- Объяснения ----------
    print("Генерация обоснований...")
    explanations = generate_explanations(snapshot, recommendations, profiles, rec_df)

    # Сохранение итогового файла
    output_file = output_dir / f'recommendations_{target_date_str}.csv'
    explanations.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Готово. Результат сохранён в {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='Целевой месяц в формате YYYY-MM-01')
    parser.add_argument('--max-k', type=int, default=10, help='Максимальное число кластеров для перебора')
    args = parser.parse_args()
    main(args.date)