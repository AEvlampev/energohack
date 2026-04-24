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
    # Анализируем по историческим данным: для каждого факта меры смотрим, были ли оплаты в следующем месяце.
    # Строим таблицу: account_id, date_measure, measure, cluster, paid_in_next_month, payment_amount
    # Это требует более сложной обработки, здесь дадим упрощённые оценки на основе статистики.
    # Для демонстрации вернём случайные разумные значения.
    # В реальности нужно объединить payments и меры, вычислить отклик.
    # Пусть для каждого кластера вероятность зависит от типа меры и кластера.
    # Здесь заполняем фиктивные данные, но с логикой: чем выше долг и возраст, тем ниже вероятность.
    # Заглушка, которую нужно заменить на реальные расчёты.
    # В финальном решении используем реальные данные из data['payments'] и data[measure].
    avg_debt_by_cluster = snapshot.groupby('cluster')['debt_amount'].mean().to_dict()

    effect = {}
    for m in measures_list:
        probs = {}
        for cl in sorted(snapshot['cluster'].unique()):
            # Вероятность обратно пропорциональна возрасту долга и сумме (условно)
            age_factor = 1 / (1 + snapshot[snapshot['cluster'] == cl]['debt_age'].mean())
            prob = max(0.1, min(0.8, 0.5 * age_factor))
            probs[cl] = prob
        effect[m] = pd.DataFrame({'cluster': list(probs.keys()), 'prob_payment': list(probs.values())})
        effect[m]['avg_payment'] = effect[m]['cluster'].map(avg_debt_by_cluster) * 0.3  # 30% долга
    return effect