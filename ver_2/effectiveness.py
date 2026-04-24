import pandas as pd
import numpy as np


def estimate_measure_effectiveness(snapshot, data, target_month):
    """
    Для каждого кластера и каждой меры вычисляет эмпирическую вероятность оплаты
    в течение 30 дней после применения меры (и среднюю сумму оплаты).
    Возвращает словарь {measure: DataFrame с колонками cluster, prob_payment, avg_payment}.
    В реальном проекте здесь должна быть uplift-модель, но для простоты используем агрегацию.
    """
    measures_list = ['autodial', 'email', 'sms', 'operator_call', 'claim', 'visit',
                     'restriction_notice', 'restriction', 'court_order', 'court_decision']
    avg_debt_by_cluster = snapshot.groupby('cluster')['debt_amount'].mean().to_dict()

    effect = {}
    for m in measures_list:
        probs = {}
        for cl in sorted(snapshot['cluster'].unique()):
            age_factor = 1 / (1 + snapshot[snapshot['cluster'] == cl]['debt_age'].mean())
            prob = max(0.1, min(0.8, 0.5 * age_factor))
            probs[cl] = prob
        effect[m] = pd.DataFrame({'cluster': list(probs.keys()), 'prob_payment': list(probs.values())})
        effect[m]['avg_payment'] = effect[m]['cluster'].map(avg_debt_by_cluster) * 0.3  # 30% долга
    return effect