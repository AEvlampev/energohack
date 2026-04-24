import pandas as pd

def generate_explanations(snapshot, recommendations, cluster_profiles, rec_df):
    explanations = []
    for acc_id, measures in recommendations.items():
        row = snapshot[snapshot['account_id'] == acc_id].iloc[0]
        cl = row['cluster']

        # Получаем прогнозные данные
        rec_row = rec_df[rec_df['account_id'] == acc_id]
        if not rec_row.empty:
            rate = rec_row['predicted_rate'].values[0]
            recovery = rec_row['expected_recovery'].values[0]
        else:
            rate = None
            recovery = None

        cluster_info = cluster_profiles.loc[cl] if cl in cluster_profiles.index else None
        desc = f"Клиент {acc_id} в кластере {cl}. "
        if cluster_info is not None:
            desc += f"Средний долг в кластере: {cluster_info['debt_amount']:.0f} руб., возраст: {cluster_info['debt_age']:.1f} мес. "
        if rate is not None:
            desc += f"Прогноз доли возврата: {rate:.1%}, ожидаемая сумма: {recovery:.2f} руб. "
        desc += "Рекомендованные меры: " + ", ".join(measures) + "."

        explanations.append({
            'account_id': acc_id,
            'cluster': cl,
            'predicted_rate': rate,
            'expected_recovery': recovery,
            'measures': ', '.join(measures),
            'explanation': desc
        })

    return pd.DataFrame(explanations)