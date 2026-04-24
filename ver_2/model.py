import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)

def train_regression_model(snapshot, feature_cols, target_col='recovery_rate'):
    """Регрессия: предсказание доли возврата."""
    df = snapshot.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    acc_10pct = np.mean(np.abs(y - y_pred) < 0.1)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'accuracy_within_10pct': acc_10pct
    }

    model.fit(X, y)
    return model, metrics

def train_classification_model(snapshot, feature_cols, target_col='paid_next_month'):
    """Классификация: предсказание факта оплаты."""
    df = snapshot.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')

    report = classification_report(y, y_pred, output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score']
    }

    model.fit(X, y)
    return model, metrics

def predict_recovery_rate(model, df, feature_cols):
    """Прогноз доли возврата (0..1)."""
    X = df[feature_cols].fillna(0)
    preds = model.predict(X)
    return np.clip(preds, 0.0, 1.0)

def predict_payment_probability(model, df, feature_cols):
    """Прогноз вероятности оплаты (0..1)."""
    X = df[feature_cols].fillna(0)
    return model.predict_proba(X)[:, 1]