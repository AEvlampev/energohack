import pandas as pd

def generate_explanations(snapshot, recommendations, cluster_profiles, rec_df):
    explanations = []
    for acc_id, measures in recommendations.items():
        row = snapshot[snapshot['account_id'] == acc_id].iloc[0]
        cl = row['cluster']
        cluster_info = cluster_profiles.loc[cl] if cl in cluster_profiles.index else None

        profile_desc = f"Клиент {acc_id} входит в кластер {cl}."
        if cluster_info is not None:
            profile_desc += f" Средний долг в кластере: {cluster_info['debt_amount']:.0f} руб.,"
            profile_desc += f" возраст долга: {cluster_info['debt_age']:.1f} мес."

        prob_row = rec_df[rec_df['account_id'] == acc_id]
        if not prob_row.empty:
            rate = prob_row['predicted_rate'].values[0]
            recovery = prob_row['expected_recovery'].values[0]
            rate_str = f"Прогнозируемая доля возврата: {rate:.1%}"
            recovery_str = f"Ожидаемый возврат: {recovery:.2f} руб. ({rate*100:.1f}% от долга)"
        else:
            rate_str = "Прогноз недоступен"
            recovery_str = ""

        measure_reasons = []
        for m in measures:
            reason_map = {
                'email': 'автоматическая рассылка',
                'sms': 'SMS-напоминание',
                'autodial': 'автодозвон',
                'operator_call': 'звонок оператора',
                'claim': 'почтовая претензия',
                'visit': 'выезд к абоненту',
                'restriction_notice': 'уведомление об ограничении',
                'restriction': 'ограничение подачи э/э',
                'court_order': 'заявление в суд',
                'court_decision': 'получение судебного приказа'
            }
            reason = reason_map.get(m, m)
            measure_reasons.append(f"{m}: {reason}")

        full = f"{profile_desc} {rate_str}. {recovery_str}. Рекомендации: {'; '.join(measure_reasons)}."
        explanations.append({
            'account_id': acc_id,
            'explanation': full,
            'predicted_rate': rate if not prob_row.empty else None,
            'expected_recovery': recovery if not prob_row.empty else None
        })
    return pd.DataFrame(explanations)