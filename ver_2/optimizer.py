import pandas as pd
import numpy as np

def build_assignments(snapshot, effectiveness, measure_criteria, model=None, feature_cols=None):
    """
    Векторизованно создаёт таблицу допустимых назначений.
    profit = predicted_recovery_rate * debt_amount  (ожидаемый возврат)
    """
    df = snapshot.set_index('account_id', drop=False)

    # 1. Прогноз доли возврата
    if model is not None and feature_cols is not None:
        from model import predict_recovery_rate
        df['predicted_recovery_rate'] = predict_recovery_rate(model, df, feature_cols)
    else:
        df['predicted_recovery_rate'] = 0.5  # fallback

    # 2. Ожидаемый возврат в рублях
    df['expected_recovery'] = df['debt_amount'] * df['predicted_recovery_rate']

    assignments_list = []

    for m, crit in measure_criteria.items():
        if m in ['court_order_6_4k', 'court_order_11_2k', 'court_order_11_less2k']:
            continue

        mask = pd.Series(True, index=df.index)

        # Критерии срока и суммы
        if 'min_age' in crit:
            mask &= df['debt_age'] >= crit['min_age']
        if 'max_age' in crit:
            mask &= df['debt_age'] <= crit['max_age']
        if 'min_amount' in crit:
            mask &= df['debt_amount'] >= crit['min_amount']
        if 'max_amount' in crit:
            mask &= df['debt_amount'] <= crit['max_amount']

        # Условия для court_order (объединяем несколько вариантов)
        if m == 'court_order':
            cond1 = (df['debt_age'] >= 4) & (df['debt_amount'] >= 10000)
            cond2 = (df['debt_age'] >= 6) & (df['debt_amount'] >= 4000)
            cond3 = (df['debt_age'] >= 11) & (df['debt_amount'] >= 2000)
            mask &= (cond1 | cond2 | cond3)

        # Требование контактов
        if crit.get('require_contact', False):
            if m == 'sms':
                mask &= df.get('has_mobile', False).astype(bool)
            elif m in ['autodial', 'operator_call']:
                mask &= df.get('has_phone', False).astype(bool)
            elif m == 'email':
                mask &= df.get('has_email', False).astype(bool)

        # Этапность
        if m in ['restriction_notice', 'restriction']:
            mask &= df['has_info_measures'] == True
        if m in ['court_order', 'court_decision']:
            mask &= df['has_restriction_measures'] == True

        eligible = df.loc[mask]
        if len(eligible) == 0:
            continue

        rate = eligible['predicted_recovery_rate'].values
        recovery = eligible['expected_recovery'].values

        temp_df = pd.DataFrame({
            'account_id': eligible['account_id'].values,
            'measure': m,
            'profit': recovery,                      # для сортировки
            'predicted_rate': rate,
            'expected_recovery': recovery
        })
        assignments_list.append(temp_df)

    if not assignments_list:
        return pd.DataFrame(columns=['account_id', 'measure', 'profit', 'predicted_rate', 'expected_recovery'])

    return pd.concat(assignments_list, ignore_index=True)


def greedy_optimize(snapshot, effectiveness, monthly_limits, measure_criteria, model=None, feature_cols=None):
    """
    Жадное назначение мер с учётом лимитов и этапности.
    Возвращает:
        recommendations (dict): {account_id: [measure1, measure2]}
        rec_df (DataFrame): предсказанные доли возврата и ожидаемый возврат
        limits_usage (dict): {measure: {'used': int, 'limit': int}} для мер с конечным лимитом
    """
    assignments = build_assignments(snapshot, effectiveness, measure_criteria, model, feature_cols)
    if assignments.empty:
        return {}, pd.DataFrame(), {}

    assignments = assignments.sort_values('profit', ascending=False)

    # Только меры с конечным лимитом
    limits = {m: limit for m, limit in monthly_limits.items() if limit < float('inf')}
    current_usage = {m: 0 for m in limits}
    client_measure_count = {}
    recommendations = {}
    result_data = {}

    for _, row in assignments.iterrows():
        acc_id = row['account_id']
        measure = row['measure']
        rate = row['predicted_rate']
        recovery = row['expected_recovery']

        if measure in limits and current_usage[measure] >= limits[measure]:
            continue
        count = client_measure_count.get(acc_id, 0)
        if count >= 2:
            continue

        if acc_id not in recommendations:
            recommendations[acc_id] = []
        recommendations[acc_id].append(measure)

        if acc_id not in result_data:
            result_data[acc_id] = {'predicted_rate': rate, 'expected_recovery': recovery}
        else:
            if rate > result_data[acc_id]['predicted_rate']:
                result_data[acc_id]['predicted_rate'] = rate
                result_data[acc_id]['expected_recovery'] = recovery

        client_measure_count[acc_id] = count + 1
        if measure in limits:
            current_usage[measure] += 1

    # DataFrame с прогнозами
    rec_df = pd.DataFrame([
        {'account_id': k,
         'predicted_rate': v['predicted_rate'],
         'expected_recovery': v['expected_recovery']}
        for k, v in result_data.items()
    ])
    rec_df['measures'] = rec_df['account_id'].map(lambda x: ','.join(recommendations.get(x, [])))

    limits_usage = {m: {'used': current_usage[m], 'limit': limits[m]} for m in limits}

    return recommendations, rec_df, limits_usage


def optimize_measures(snapshot, effectiveness, monthly_limits, measure_criteria, model=None, feature_cols=None):
    return greedy_optimize(snapshot, effectiveness, monthly_limits, measure_criteria, model, feature_cols)