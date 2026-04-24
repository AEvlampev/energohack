import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_clustering_features(snapshot):
    """
    Возвращает матрицу признаков для кластеризации (нормированные данные) и список имён признаков.
    """
    numeric_cols = ['debt_amount', 'debt_age', 'avg_payment_3m', 'avg_payment_6m',
                    'avg_accrued_3m', 'payment_ratio', 'num_contacts', 'trend_debt']
    bool_cols = ['has_phone', 'has_email', 'has_mobile', 'has_benefits', 'gasification',
                 'not_living', 'mkd', 'dormitory', 'remote_disconnect']
    measure_cols = [f'prev_{m}' for m in ['autodial', 'email', 'sms', 'operator_call', 'claim', 'visit',
                                          'restriction_notice', 'restriction', 'court_order', 'court_decision']]

    available_cols = [c for c in numeric_cols + bool_cols + measure_cols if c in snapshot.columns]
    df_features = snapshot[available_cols].copy()

    df_features = df_features.fillna(0)

    num_feat = [c for c in numeric_cols if c in df_features.columns]
    bool_feat = [c for c in bool_cols + measure_cols if c in df_features.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feat),
        ('bool', 'passthrough', bool_feat)
    ])

    X = preprocessor.fit_transform(df_features)
    feature_names = num_feat + bool_feat
    return X, feature_names, preprocessor


def add_cluster_labels(snapshot, labels):
    snapshot = snapshot.copy()
    snapshot['cluster'] = labels
    return snapshot