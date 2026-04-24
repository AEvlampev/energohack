import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

def load_all_data():
    data = {}
    base = Path(cfg.DATA_PATH)

    df_info = pd.read_excel(base / cfg.CONFIG_FILES['info'])
    column_mapping = {
        'ЛС': 'account_id',
        'Возможность дистанционного отключения': 'remote_disconnect',
        'Наличие телефона': 'has_phone',
        'Наличие льгот': 'has_benefits',
        'Газификация дома': 'gasification',
        'Город': 'city',
        'ЯрОблИЕРЦ квитанция': 'yar_obl_receipt',
        'Почта России квитанция': 'post_receipt',
        'электронная квитанция': 'email_receipt',
        'не проживает': 'not_living',
        'ЧД': 'chd',
        'МКД': 'mkd',
        'Общежитие': 'dormitory',
        'Установка Тамбур': 'tambour',
        'Установка опора': 'support',
        'Установка в квартире/доме': 'flat_install',
        'Установка лестничкая клетка': 'staircase',
        'Адрес (ГУИД)': 'guid'
    }
    df_info = df_info.rename(columns=column_mapping)
    df_info['account_id'] = pd.to_numeric(df_info['account_id'], errors='coerce')
    df_info = df_info.dropna(subset=['account_id'])
    df_info['account_id'] = df_info['account_id'].astype(int)
    data['info'] = df_info

    df_raw = pd.read_excel(base / cfg.CONFIG_FILES['turnover'], header=None)
    num_columns = df_raw.shape[1]

    dates_row = df_raw.iloc[0]
    columns = ['account_id']
    i = 1
    while i < num_columns:
        dt_val = dates_row[i]
        if pd.isna(dt_val):
            break
        dt = pd.to_datetime(dt_val, errors='coerce')
        if pd.isna(dt):
            break
        dt_str = dt.strftime("%Y-%m-%d")
        columns.extend([
            f'{dt_str}_opening_balance',
            f'{dt_str}_accrued',
            f'{dt_str}_paid'
        ])
        i += 3

    columns = columns[:num_columns]

    seen = {}
    unique_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            unique_columns.append(col)
        else:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")
    columns = unique_columns

    df_data = df_raw.iloc[2:, :num_columns].copy()
    df_turnover = pd.DataFrame(df_data.values, columns=columns)

    df_turnover['account_id'] = pd.to_numeric(df_turnover['account_id'], errors='coerce')
    df_turnover = df_turnover.dropna(subset=['account_id'])
    df_turnover['account_id'] = df_turnover['account_id'].astype(int)

    for col in df_turnover.columns:
        if col != 'account_id':
            df_turnover[col] = pd.to_numeric(df_turnover[col], errors='coerce').fillna(0)

    data['turnover'] = df_turnover

    df_payments = pd.read_csv(base / cfg.CONFIG_FILES['payments'], sep=';', decimal=',')
    df_payments.columns = ['account_id', 'payment_date', 'amount', 'method']
    df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'], dayfirst=True, errors='coerce')
    df_payments['account_id'] = pd.to_numeric(df_payments['account_id'], errors='coerce')
    df_payments = df_payments.dropna(subset=['account_id', 'payment_date'])
    df_payments['account_id'] = df_payments['account_id'].astype(int)
    data['payments'] = df_payments

    measure_files = [
        'autodial', 'email', 'sms', 'operator_call', 'claim', 'visit',
        'restriction_notice', 'restriction', 'court_order', 'court_decision'
    ]
    for m in measure_files:
        df_raw = pd.read_excel(base / cfg.CONFIG_FILES[m], header=None)
        header_idx = None
        for idx, row in df_raw.iterrows():
            row_str = [str(cell).strip() for cell in row if isinstance(cell, str)]
            if 'ЛС' in row_str and 'Дата' in row_str:
                header_idx = idx
                break
        if header_idx is None:
            header_idx = 0

        df = df_raw.iloc[header_idx + 1:].copy()
        df.columns = ['account_id', 'date']
        df = df.iloc[:, :2]

        df['account_id'] = pd.to_numeric(df['account_id'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['account_id', 'date'])
        df['account_id'] = df['account_id'].astype(int)
        data[m] = df

    df_limits = pd.read_excel(base / cfg.CONFIG_FILES['limits'])
    data['limits'] = df_limits

    return data


def build_monthly_snapshot(data, target_month):
    """
    Формирует срез должников на начало target_month.
    Возвращает DataFrame с признаками и целевой переменной recovery_rate (доля возврата долга).
    """
    turnover = data['turnover'].copy()
    id_vars = ['account_id']
    value_vars = [c for c in turnover.columns if c != 'account_id']
    df_long = pd.melt(turnover, id_vars=id_vars, value_vars=value_vars,
                      var_name='temp', value_name='value')
    df_long[['date', 'metric']] = df_long['temp'].str.extract(r'(\d{4}-\d{2}-\d{2})_(.+)')
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long = df_long.drop('temp', axis=1)
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_pivot = df_long.pivot_table(index=['account_id', 'date'], columns='metric', values='value',
                                   aggfunc='first').reset_index()

    current = df_pivot[df_pivot['date'] == target_month][['account_id', 'opening_balance']]
    current = current.rename(columns={'opening_balance': 'debt_opening'})

    info = data['info'].copy()
    bool_cols = ['remote_disconnect', 'has_phone', 'has_benefits', 'gasification',
                 'yar_obl_receipt', 'post_receipt', 'email_receipt', 'not_living',
                 'chd', 'mkd', 'dormitory', 'tambour', 'support', 'flat_install', 'staircase']
    for c in bool_cols:
        if c in info.columns:
            info[c] = info[c].map({'Да': True, 'Нет': False}).fillna(False)
    info['has_email'] = info['email_receipt']
    info['has_mobile'] = info['has_phone']

    snapshot = current.merge(info, on='account_id', how='left')

    history = df_pivot[df_pivot['date'] < target_month].sort_values('date')
    if not history.empty:
        hist_agg = history.groupby('account_id').agg(
            avg_payment_3m=('paid', lambda x: x.tail(3).mean() if len(x)>=3 else x.mean()),
            avg_payment_6m=('paid', lambda x: x.tail(6).mean() if len(x)>=6 else x.mean()),
            avg_accrued_3m=('accrued', lambda x: x.tail(3).mean() if len(x)>=3 else x.mean()),
            payment_ratio=('paid', lambda x: (x.sum() / x.tail(6).shape[0]) if len(x) else 0),
            months_with_debt=('opening_balance', lambda x: (x > 0).sum()),
            trend_debt=('opening_balance', lambda x: np.polyfit(range(len(x)), x.fillna(0), 1)[0] if len(x)>=3 else 0),
            # Последний платёж и долг в последнем месяце (для расчёта recovery_rate)
            last_payment_amount=('paid', lambda x: x.iloc[-1] if len(x) > 0 else 0),
            last_opening_balance=('opening_balance', lambda x: x.iloc[-1] if len(x) > 0 else np.nan)
        ).reset_index()
        snapshot = snapshot.merge(hist_agg, on='account_id', how='left')
    else:
        for col in ['avg_payment_3m', 'avg_payment_6m', 'avg_accrued_3m', 'payment_ratio',
                    'months_with_debt', 'trend_debt', 'last_payment_amount', 'last_opening_balance']:
            snapshot[col] = 0.0

    snapshot['debt_age'] = snapshot['months_with_debt'].fillna(0)
    snapshot['debt_amount'] = snapshot['debt_opening'].clip(lower=0)
    snapshot['num_contacts'] = (snapshot['has_phone'].astype(int) +
                                snapshot['has_email'].astype(int) +
                                snapshot['has_mobile'].astype(int))

    for m in ['autodial','email','sms','operator_call','claim','visit',
              'restriction_notice','restriction','court_order','court_decision']:
        if m in data:
            measures = data[m]
            applied = measures[measures['date'] < target_month].groupby('account_id').size().gt(0).astype(int)
            snapshot[f'prev_{m}'] = snapshot['account_id'].map(applied).fillna(0).astype(bool)

    snapshot = snapshot.fillna(0)

    info_measures = ['prev_autodial','prev_email','prev_sms','prev_operator_call','prev_claim','prev_visit']
    restriction_measures = ['prev_restriction_notice','prev_restriction']
    snapshot['has_info_measures'] = snapshot[info_measures].any(axis=1)
    snapshot['has_restriction_measures'] = snapshot[restriction_measures].any(axis=1)

    snapshot['recovery_rate'] = np.where(
        snapshot['last_opening_balance'] > 0,
        np.minimum(snapshot['last_payment_amount'] / snapshot['last_opening_balance'], 1.0),
        0.0
    )
    snapshot = snapshot.dropna(subset=['recovery_rate'])

    snapshot = snapshot.drop_duplicates(subset='account_id', keep='first')

    return snapshot