import os

import pandas as pd

# Пути к файлам
DATA_PATH = os.path.join(str(os.path.dirname(__file__)), "..", 'data')
CONFIG_FILES = {
    'info': '01 Общая информация о ЛС ХК.xlsx',
    'turnover': '02 Обортно-сальдовая ведомость ЛС ХК.xlsx',
    'payments': '03 Оплаты ХК.csv',
    'autodial': '04 Автодозвон ХК.xlsx',
    'email': '05 E-mail ХК.xlsx',
    'sms': '06 СМС ХК.xlsx',
    'operator_call': '07 Обзвон оператором ХК.xlsx',
    'claim': '08 Претензия ХК.xlsx',
    'visit': '09 Выезд к абоненту ХК.xlsx',
    'restriction_notice': '10 Уведомление о введении ограничения ХК.xlsx',
    'restriction': '11 Ограничение ХК.xlsx',
    'court_order': '12 Заявление о выдаче судебного приказа ХК.xlsx',
    'court_decision': '13 Получение судебного приказа или ИЛ ХК.xlsx',
    'limits': '14 Лимиты мер воздействия ХК.xlsx',
}

# Критерии допустимости мер (Приложение 1) – срок долга, сумма, доп.условия
MEASURE_CRITERIA = {
    'email': {'min_age': 1, 'min_amount': 500, 'require_contact': True},
    'sms': {'min_age': 1, 'max_age': 2, 'min_amount': 500, 'max_amount': 2000, 'require_contact': True},
    'autodial': {'min_age': 2, 'max_age': 6, 'min_amount': 500, 'max_amount': 1500, 'require_contact': True},
    'operator_call': {'min_age': 2, 'max_age': 6, 'min_amount': 1500, 'require_contact': True},
    'claim': {'min_age': 2, 'max_age': 6, 'min_amount': 500, 'require_contact': False,  # доп.условия: нет телефона/недозвон/отказ
              'contact_condition': 'no_phone_or_refusal'},
    'restriction_notice': {'min_age': 1, 'min_amount': 1500, 'require_info_fail': True,
                           'require_debt_gt_double_normative': True},  # упрощено
    'court_order': {'min_age': 4, 'min_amount': 10000, 'require_restriction_fail': True},
    'court_order_6_4k': {'min_age': 6, 'min_amount': 4000, 'require_restriction_fail': True},
    'court_order_11_2k': {'min_age': 11, 'min_amount': 2000, 'require_restriction_fail': True},
    'court_order_11_less2k': {'min_age': 11, 'max_amount': 2000, 'require_commission': True},  # особая комиссия
}

# Месячные лимиты (Приложение 2)
MONTHLY_LIMITS = {
    'autodial': 8000,
    'email': float('inf'),
    'sms': 2150,
    'operator_call': 1550,
    'claim': 400,
    'visit': 500,
    'restriction_notice': 6200,
    'restriction': 200,
    'court_order': 400,
    'court_decision': 250,
}

# Этапы процедур (1 – информирование, 2 – ограничение, 3 – взыскание)
STAGES = {
    'email': 1,
    'sms': 1,
    'autodial': 1,
    'operator_call': 1,
    'claim': 1,
    'visit': 1,  # условно отнесём к информированию, но выезд может быть и на других этапах
    'restriction_notice': 2,
    'restriction': 2,
    'court_order': 3,
    'court_decision': 3,
}