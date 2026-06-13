# ============================================================
# BKASH CHURN PREDICTION - MAXIMUM AUC PIPELINE v4
# Target: surpass 0.9985 AUC
# ============================================================
import os, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime, date

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import linregress, rankdata, entropy

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ============================================================
# PATHS
# ============================================================
ROOT       = Path('/kaggle/input/datasets/umor2525/bikash/public')
TRX_DIR    = ROOT / 'transactions'
BAL_DIR    = ROOT / 'dayend_balance'
KYC_PATH   = ROOT / 'kyc.parquet'
TRAIN_PATH = ROOT / 'train_labels.csv'
TEST_PATH  = ROOT / 'test.csv'

OBS_START = date(2024, 1, 1)
OBS_END   = date(2024, 3, 31)
REF_DATE  = date(2024, 4, 1)
MONTHS    = ['2024-01', '2024-02', '2024-03']
TRX_TYPES = ['P2P', 'MerchantPay', 'BillPay', 'CashIn', 'CashOut']

# ============================================================
# STEP 1: LOAD TARGET ACCOUNTS
# ============================================================
print("Loading labels and test accounts...")
train_labels = pd.read_csv(TRAIN_PATH)
test_df      = pd.read_csv(TEST_PATH)

all_accounts      = set(train_labels['ACCOUNT_ID'].tolist() + test_df['ACCOUNT_ID'].tolist())
all_accounts_list = list(all_accounts)
print(f"Total target accounts: {len(all_accounts_list):,}")

# ============================================================
# STEP 2: LOAD ALL TRANSACTIONS (SRC side)
# ============================================================
print("\nProcessing transactions (SRC side)...")

def load_trx_file(filepath: Path, account_set: set) -> pl.DataFrame:
    df = pl.read_parquet(filepath)
    df = df.with_columns([
        pl.col('TRX_DATETIME')
          .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%.f', strict=False)
          .alias('TRX_DATETIME')
    ])
    df = df.filter(
        (pl.col('TRX_DATETIME').dt.date() >= OBS_START) &
        (pl.col('TRX_DATETIME').dt.date() <= OBS_END)
    )
    df = df.with_columns([
        pl.col('TRX_DATETIME').dt.date().alias('TRX_DATE'),
        pl.col('TRX_DATETIME').dt.weekday().alias('DOW'),
        pl.col('TRX_DATETIME').dt.hour().alias('HOUR'),
        pl.col('TRX_DATETIME').dt.month().alias('MONTH'),
        pl.col('TRX_DATETIME').dt.week().alias('WEEK_NUM'),
        pl.col('TRX_DATETIME').dt.ordinal_day().alias('DAY_OF_YEAR'),
    ])
    account_series = pl.Series('_tmp', list(account_set))
    df_src = df.filter(pl.col('SRC_ACCOUNT').is_in(account_series))
    df_src = df_src.rename({'SRC_ACCOUNT': 'ACCOUNT_ID'})
    return df_src

trx_frames = []
for month in MONTHS:
    fp = TRX_DIR / f'trx_{month}.parquet'
    print(f"  Loading {fp.name}...")
    trx_frames.append(load_trx_file(fp, all_accounts))
    gc.collect()

trx_all = pl.concat(trx_frames)
del trx_frames
gc.collect()
print(f"  Filtered SRC transactions: {len(trx_all):,}")

# ============================================================
# FEATURE BLOCK A: Global RFM
# ============================================================
print("Building RFM features...")

last_trx = (
    trx_all.group_by('ACCOUNT_ID')
    .agg(pl.col('TRX_DATE').max().alias('LAST_TRX_DATE'))
    .with_columns([
        (pl.lit(REF_DATE) - pl.col('LAST_TRX_DATE'))
          .dt.total_days().alias('RECENCY_DAYS')
    ])
)

global_agg = (
    trx_all.group_by('ACCOUNT_ID')
    .agg([
        pl.len().alias('TRX_COUNT'),
        pl.col('TRX_AMT').sum().alias('TRX_AMT_SUM'),
        pl.col('TRX_AMT').mean().alias('TRX_AMT_MEAN'),
        pl.col('TRX_AMT').std().alias('TRX_AMT_STD'),
        pl.col('TRX_AMT').max().alias('TRX_AMT_MAX'),
        pl.col('TRX_AMT').min().alias('TRX_AMT_MIN'),
        pl.col('TRX_AMT').median().alias('TRX_AMT_MEDIAN'),
        pl.col('TRX_DATE').n_unique().alias('ACTIVE_DAYS'),
        pl.col('DST_ACCOUNT').n_unique().alias('UNIQUE_DST'),
        pl.col('DOW').mean().alias('AVG_DOW'),
        pl.col('DOW').std().alias('STD_DOW'),
        pl.col('HOUR').mean().alias('AVG_HOUR'),
        pl.col('HOUR').std().alias('STD_HOUR'),
        pl.col('TRX_TYPE').n_unique().alias('UNIQUE_TRX_TYPES'),
        pl.col('TRX_AMT').quantile(0.25).alias('TRX_AMT_Q25'),
        pl.col('TRX_AMT').quantile(0.75).alias('TRX_AMT_Q75'),
        (pl.col('DOW') >= 5).sum().alias('WEEKEND_TRX_COUNT'),
        ((pl.col('HOUR') >= 22) | (pl.col('HOUR') <= 6)).sum().alias('NIGHT_TRX_COUNT'),
        pl.col('WEEK_NUM').n_unique().alias('ACTIVE_WEEKS'),
        pl.col('WEEK_NUM').max().alias('LAST_ACTIVE_WEEK'),
        pl.col('WEEK_NUM').min().alias('FIRST_ACTIVE_WEEK'),
    ])
)

# ============================================================
# FEATURE BLOCK B: Per-type aggregations + per-type recency
# ============================================================
print("Building per-type features...")

type_agg = (
    trx_all.group_by(['ACCOUNT_ID', 'TRX_TYPE'])
    .agg([
        pl.len().alias('TYPE_COUNT'),
        pl.col('TRX_AMT').sum().alias('TYPE_AMT'),
        pl.col('TRX_AMT').mean().alias('TYPE_AMT_MEAN'),
        pl.col('TRX_AMT').std().alias('TYPE_AMT_STD'),
        pl.col('TRX_DATE').n_unique().alias('TYPE_ACTIVE_DAYS'),
        # Per-type recency: days since last transaction of this type
        pl.col('TRX_DATE').max().alias('TYPE_LAST_DATE'),
    ])
    .with_columns([
        (pl.lit(REF_DATE) - pl.col('TYPE_LAST_DATE'))
          .dt.total_days().alias('TYPE_RECENCY_DAYS')
    ])
)

def pivot_type(df, value_col):
    p = df.pivot(on='TRX_TYPE', index='ACCOUNT_ID', values=value_col)
    p.columns = ['ACCOUNT_ID'] + [f'{value_col}_{c}' for c in p.columns[1:]]
    return p

type_count_p     = pivot_type(type_agg, 'TYPE_COUNT')
type_amt_p       = pivot_type(type_agg, 'TYPE_AMT')
type_amt_mean_p  = pivot_type(type_agg, 'TYPE_AMT_MEAN')
type_days_p      = pivot_type(type_agg, 'TYPE_ACTIVE_DAYS')
type_recency_p   = pivot_type(type_agg, 'TYPE_RECENCY_DAYS')  # NEW

# ============================================================
# FEATURE BLOCK C: Monthly trend + velocity + acceleration
# ============================================================
print("Building monthly trend features...")

monthly_agg = (
    trx_all.group_by(['ACCOUNT_ID', 'MONTH'])
    .agg([
        pl.len().alias('M_COUNT'),
        pl.col('TRX_AMT').sum().alias('M_AMT'),
        pl.col('TRX_DATE').n_unique().alias('M_ACTIVE_DAYS'),
        pl.col('TRX_AMT').mean().alias('M_AMT_MEAN'),
        pl.col('TRX_TYPE').n_unique().alias('M_UNIQUE_TYPES'),
    ])
)

mp = monthly_agg.to_pandas()
mp_pivot = mp.pivot_table(
    index='ACCOUNT_ID', columns='MONTH',
    values=['M_COUNT', 'M_AMT', 'M_ACTIVE_DAYS', 'M_AMT_MEAN', 'M_UNIQUE_TYPES']
)
mp_pivot.columns = [f'{v}_{m}' for v, m in mp_pivot.columns]
mp_pivot = mp_pivot.reset_index()

for val in ['M_COUNT', 'M_AMT', 'M_ACTIVE_DAYS', 'M_UNIQUE_TYPES']:
    c1 = f'{val}_2024-01'
    c2 = f'{val}_2024-02'
    c3 = f'{val}_2024-03'
    if all(c in mp_pivot.columns for c in [c1, c2, c3]):
        mp_pivot[f'{val}_JAN_FEB_RATIO'] = (mp_pivot[c2] + 1) / (mp_pivot[c1] + 1)
        mp_pivot[f'{val}_FEB_MAR_RATIO'] = (mp_pivot[c3] + 1) / (mp_pivot[c2] + 1)
        mp_pivot[f'{val}_JAN_MAR_RATIO'] = (mp_pivot[c3] + 1) / (mp_pivot[c1] + 1)
        mp_pivot[f'{val}_SLOPE'] = mp_pivot[[c1, c2, c3]].apply(
            lambda r: linregress([1, 2, 3], r.fillna(0).values)[0], axis=1
        )
        mp_pivot[f'{val}_ACCEL'] = (
            (mp_pivot[c3] - mp_pivot[c2]) - (mp_pivot[c2] - mp_pivot[c1])
        )

# ============================================================
# FEATURE BLOCK D: Recency windows (7/14/30/60 days)
# ============================================================
print("Building recency window features...")

windows = {
    'LAST_7D':  date(2024, 3, 25),
    'LAST_14D': date(2024, 3, 18),
    'LAST_30D': date(2024, 3,  2),
    'LAST_60D': date(2024, 2,  1),
}

window_dfs = []
for name, start in windows.items():
    w = (
        trx_all.filter(pl.col('TRX_DATE') >= start)
        .group_by('ACCOUNT_ID')
        .agg([
            pl.len().alias(f'{name}_COUNT'),
            pl.col('TRX_AMT').sum().alias(f'{name}_AMT'),
            pl.col('TRX_DATE').n_unique().alias(f'{name}_ACTIVE_DAYS'),
            pl.col('TRX_TYPE').n_unique().alias(f'{name}_UNIQUE_TYPES'),
        ])
    )
    window_dfs.append(w)

# ============================================================
# FEATURE BLOCK E: Weekly cadence (which weeks active + trailing silence)
# ============================================================
print("Building weekly cadence features...")

# Week-level activity: 13 weeks in Jan-Mar 2024
weekly_agg = (
    trx_all.group_by(['ACCOUNT_ID', 'WEEK_NUM'])
    .agg([
        pl.len().alias('W_COUNT'),
        pl.col('TRX_AMT').sum().alias('W_AMT'),
    ])
)

weekly_pd = weekly_agg.to_pandas()
week_nums = sorted(weekly_pd['WEEK_NUM'].unique())

weekly_pivot = weekly_pd.pivot_table(
    index='ACCOUNT_ID', columns='WEEK_NUM', values='W_COUNT', fill_value=0
)
weekly_pivot.columns = [f'WEEK_{w}_COUNT' for w in weekly_pivot.columns]
weekly_pivot = weekly_pivot.reset_index()

# Trailing inactive weeks (count of consecutive zero-activity weeks from end of March)
def trailing_inactive_weeks(row, week_cols):
    vals = row[week_cols].values[::-1]  # reverse: most recent first
    count = 0
    for v in vals:
        if v == 0:
            count += 1
        else:
            break
    return count

week_cols = [c for c in weekly_pivot.columns if c.startswith('WEEK_')]
weekly_pivot['TRAILING_INACTIVE_WEEKS'] = weekly_pivot.apply(
    lambda r: trailing_inactive_weeks(r, week_cols), axis=1
)
weekly_pivot['ACTIVE_WEEK_COUNT'] = (weekly_pivot[week_cols] > 0).sum(axis=1)
weekly_pivot['WEEK_ACTIVITY_RATE'] = weekly_pivot['ACTIVE_WEEK_COUNT'] / len(week_cols)

# ============================================================
# FEATURE BLOCK F: Exponential decay weighted amount (recent activity weight)
# ============================================================
print("Building exponential decay features...")

trx_pd_decay = trx_all.select(['ACCOUNT_ID', 'TRX_DATE', 'TRX_AMT']).to_pandas()
trx_pd_decay['TRX_DATE'] = pd.to_datetime(trx_pd_decay['TRX_DATE'])
ref_dt = pd.Timestamp(REF_DATE)
trx_pd_decay['DAYS_TO_REF'] = (ref_dt - trx_pd_decay['TRX_DATE']).dt.days

# Exponential decay weights: lambda=0.05 (half-life ~14 days)
trx_pd_decay['DECAY_WEIGHT'] = np.exp(-0.05 * trx_pd_decay['DAYS_TO_REF'])
trx_pd_decay['DECAY_AMT']    = trx_pd_decay['TRX_AMT'] * trx_pd_decay['DECAY_WEIGHT']

decay_agg = (
    trx_pd_decay.groupby('ACCOUNT_ID')
    .agg(
        DECAY_AMT_SUM=('DECAY_AMT', 'sum'),
        DECAY_WEIGHT_SUM=('DECAY_WEIGHT', 'sum'),
        DECAY_AMT_MEAN=('DECAY_AMT', 'mean'),
    )
    .reset_index()
)
# Normalized decay score: recent-weighted amount vs raw sum
decay_agg = decay_agg.merge(
    global_agg.select(['ACCOUNT_ID', 'TRX_AMT_SUM']).to_pandas(),
    on='ACCOUNT_ID', how='left'
)
decay_agg['DECAY_SCORE'] = decay_agg['DECAY_AMT_SUM'] / (decay_agg['TRX_AMT_SUM'] + 1)
del trx_pd_decay
gc.collect()

# ============================================================
# FEATURE BLOCK G: Hour entropy (behavioral regularity)
# ============================================================
print("Building entropy features...")

trx_pd_ent = trx_all.select(['ACCOUNT_ID', 'HOUR', 'DOW', 'TRX_TYPE']).to_pandas()

def safe_entropy(series):
    counts = series.value_counts()
    probs  = counts / counts.sum()
    return entropy(probs)

hour_entropy = (
    trx_pd_ent.groupby('ACCOUNT_ID')['HOUR']
    .apply(safe_entropy)
    .reset_index()
    .rename(columns={'HOUR': 'HOUR_ENTROPY'})
)
dow_entropy = (
    trx_pd_ent.groupby('ACCOUNT_ID')['DOW']
    .apply(safe_entropy)
    .reset_index()
    .rename(columns={'DOW': 'DOW_ENTROPY'})
)
type_entropy = (
    trx_pd_ent.groupby('ACCOUNT_ID')['TRX_TYPE']
    .apply(safe_entropy)
    .reset_index()
    .rename(columns={'TRX_TYPE': 'TYPE_ENTROPY'})
)
del trx_pd_ent
gc.collect()

# ============================================================
# FEATURE BLOCK H: CashIn vs CashOut net flow
# ============================================================
print("Building cash flow features...")

cashin_agg = (
    trx_all.filter(pl.col('TRX_TYPE') == 'CashIn')
    .group_by('ACCOUNT_ID')
    .agg([
        pl.col('TRX_AMT').sum().alias('CASHIN_SUM'),
        pl.len().alias('CASHIN_COUNT'),
    ])
)
cashout_agg = (
    trx_all.filter(pl.col('TRX_TYPE') == 'CashOut')
    .group_by('ACCOUNT_ID')
    .agg([
        pl.col('TRX_AMT').sum().alias('CASHOUT_SUM'),
        pl.len().alias('CASHOUT_COUNT'),
    ])
)

# ============================================================
# FEATURE BLOCK I: Network / DST side
# ============================================================
print("Building network features (DST side)...")

trx_dst_frames = []
for month in MONTHS:
    fp = TRX_DIR / f'trx_{month}.parquet'
    df = pl.read_parquet(fp, columns=['DST_ACCOUNT', 'SRC_ACCOUNT', 'TRX_AMT', 'TRX_TYPE', 'TRX_DATETIME'])
    df = df.with_columns([
        pl.col('TRX_DATETIME')
          .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%.f', strict=False)
          .alias('TRX_DATETIME')
    ])
    df = df.filter(
        (pl.col('TRX_DATETIME').dt.date() >= OBS_START) &
        (pl.col('TRX_DATETIME').dt.date() <= OBS_END)
    )
    account_series = pl.Series('_tmp', all_accounts_list)
    df_dst = df.filter(pl.col('DST_ACCOUNT').is_in(account_series))
    df_dst = df_dst.rename({'DST_ACCOUNT': 'ACCOUNT_ID'})
    trx_dst_frames.append(df_dst)
    gc.collect()

trx_dst_all = pl.concat(trx_dst_frames)
del trx_dst_frames
gc.collect()

incoming_agg = (
    trx_dst_all.group_by('ACCOUNT_ID')
    .agg([
        pl.len().alias('INCOMING_COUNT'),
        pl.col('TRX_AMT').sum().alias('INCOMING_AMT'),
        pl.col('TRX_AMT').mean().alias('INCOMING_AMT_MEAN'),
        pl.col('SRC_ACCOUNT').n_unique().alias('UNIQUE_SRC'),
    ])
)
del trx_dst_all
gc.collect()

merchant_agg = (
    trx_all.filter(pl.col('TRX_TYPE') == 'MerchantPay')
    .group_by('ACCOUNT_ID')
    .agg([
        pl.col('DST_ACCOUNT').n_unique().alias('UNIQUE_MERCHANTS'),
        pl.col('TRX_AMT').sum().alias('MERCHANT_AMT_SUM'),
        pl.col('TRX_AMT').mean().alias('MERCHANT_AMT_MEAN'),
    ])
)

# ============================================================
# FEATURE BLOCK J: Inter-transaction gap statistics
# ============================================================
print("Building gap features...")

trx_sorted    = trx_all.sort(['ACCOUNT_ID', 'TRX_DATE'])
trx_with_prev = trx_sorted.with_columns([
    pl.col('TRX_DATE').shift(1).over('ACCOUNT_ID').alias('PREV_DATE')
]).with_columns([
    (pl.col('TRX_DATE') - pl.col('PREV_DATE')).dt.total_days().alias('GAP_DAYS')
])

gap_agg = (
    trx_with_prev.filter(pl.col('GAP_DAYS').is_not_null())
    .group_by('ACCOUNT_ID')
    .agg([
        pl.col('GAP_DAYS').mean().alias('GAP_MEAN'),
        pl.col('GAP_DAYS').std().alias('GAP_STD'),
        pl.col('GAP_DAYS').max().alias('GAP_MAX'),
        pl.col('GAP_DAYS').median().alias('GAP_MEDIAN'),
        pl.col('GAP_DAYS').quantile(0.9).alias('GAP_P90'),
    ])
)
del trx_sorted, trx_with_prev
gc.collect()

# ============================================================
# FEATURE BLOCK K: Balance features
# ============================================================
print("Building balance features...")

bal_frames = []
for month in MONTHS:
    fp = BAL_DIR / f'balance_{month}.parquet'
    print(f"  Loading {fp.name}...")
    df = pl.read_parquet(fp)
    account_series = pl.Series('_tmp', all_accounts_list)
    df = df.filter(pl.col('ACCOUNT_ID').is_in(account_series))
    bal_frames.append(df)
    gc.collect()

bal_all = pl.concat(bal_frames)
del bal_frames
gc.collect()
print(f"  Filtered balance rows: {len(bal_all):,}")

bal_agg = (
    bal_all.sort(['ACCOUNT_ID', 'DATE'])
    .group_by('ACCOUNT_ID')
    .agg([
        pl.col('AVAILABLE_BALANCE').mean().alias('BAL_MEAN'),
        pl.col('AVAILABLE_BALANCE').std().alias('BAL_STD'),
        pl.col('AVAILABLE_BALANCE').min().alias('BAL_MIN'),
        pl.col('AVAILABLE_BALANCE').max().alias('BAL_MAX'),
        pl.col('AVAILABLE_BALANCE').last().alias('BAL_LAST'),
        pl.col('AVAILABLE_BALANCE').first().alias('BAL_FIRST'),
        pl.col('AVAILABLE_BALANCE').median().alias('BAL_MEDIAN'),
        (pl.col('AVAILABLE_BALANCE') == 0).sum().alias('ZERO_BAL_DAYS'),
        (pl.col('AVAILABLE_BALANCE') < 0).sum().alias('NEG_BAL_DAYS'),
        pl.col('AVAILABLE_BALANCE').count().alias('BAL_DAYS'),
        pl.col('AVAILABLE_BALANCE').tail(7).mean().alias('BAL_LAST7_MEAN'),
        pl.col('AVAILABLE_BALANCE').tail(7).min().alias('BAL_LAST7_MIN'),
        pl.col('AVAILABLE_BALANCE').tail(14).mean().alias('BAL_LAST14_MEAN'),
        pl.col('AVAILABLE_BALANCE').head(7).mean().alias('BAL_FIRST7_MEAN'),
        # Consecutive zero balance at end (trailing zeros)
        pl.col('AVAILABLE_BALANCE').tail(14).eq(0).sum().alias('TRAILING_ZERO_BAL_14D'),
        pl.col('AVAILABLE_BALANCE').tail(7).eq(0).sum().alias('TRAILING_ZERO_BAL_7D'),
    ])
)

print("  Computing balance slope...")
bal_pd = bal_all.sort(['ACCOUNT_ID', 'DATE']).to_pandas()
bal_slope = (
    bal_pd.groupby('ACCOUNT_ID')['AVAILABLE_BALANCE']
    .apply(lambda x: linregress(np.arange(len(x)), x.values)[0] if len(x) > 1 else 0.0)
    .reset_index()
    .rename(columns={'AVAILABLE_BALANCE': 'BAL_SLOPE'})
)

bal_pd['MONTH'] = pd.to_datetime(bal_pd['DATE']).dt.month
bal_monthly = (
    bal_pd.groupby(['ACCOUNT_ID', 'MONTH'])['AVAILABLE_BALANCE']
    .mean().unstack(fill_value=np.nan)
)
bal_monthly.columns = [f'BAL_MONTH_{m}' for m in bal_monthly.columns]
bal_monthly = bal_monthly.reset_index()
bal_monthly['BAL_MONTH_SLOPE'] = bal_monthly[
    [c for c in bal_monthly.columns if 'BAL_MONTH_' in c]
].apply(lambda r: linregress([1, 2, 3], r.fillna(r.mean()).values)[0], axis=1)

del bal_pd, bal_all
gc.collect()

# ============================================================
# FEATURE BLOCK L: KYC features
# ============================================================
print("Building KYC features...")

kyc = pl.read_parquet(KYC_PATH)
account_series = pl.Series('_tmp', all_accounts_list)
kyc = kyc.filter(pl.col('ACCOUNT_ID').is_in(account_series))
kyc = kyc.with_columns([
    pl.col('ACCOUNT_OPEN_DATE').cast(pl.Date).alias('ACCOUNT_OPEN_DATE')
]).with_columns([
    (pl.lit(REF_DATE) - pl.col('ACCOUNT_OPEN_DATE')).dt.total_days().alias('ACCOUNT_AGE_DAYS')
])
kyc_pd = kyc.to_pandas()

for col in ['ACCOUNT_TYPE', 'GENDER', 'REGION']:
    le = LabelEncoder()
    kyc_pd[col] = le.fit_transform(kyc_pd[col].astype(str))

# ============================================================
# STEP 3: MERGE ALL FEATURES
# ============================================================
print("\nMerging all features...")

base = pd.DataFrame({'ACCOUNT_ID': all_accounts_list})

merge_list = [
    last_trx.to_pandas()[['ACCOUNT_ID', 'RECENCY_DAYS']],
    global_agg.to_pandas(),
    type_count_p.to_pandas(),
    type_amt_p.to_pandas(),
    type_amt_mean_p.to_pandas(),
    type_days_p.to_pandas(),
    type_recency_p.to_pandas(),       # NEW
    mp_pivot,
    incoming_agg.to_pandas(),
    merchant_agg.to_pandas(),
    gap_agg.to_pandas(),
    bal_agg.to_pandas(),
    bal_slope,
    bal_monthly,
    weekly_pivot,                      # NEW
    decay_agg[['ACCOUNT_ID', 'DECAY_AMT_SUM', 'DECAY_WEIGHT_SUM',
               'DECAY_AMT_MEAN', 'DECAY_SCORE']],  # NEW
    hour_entropy,                      # NEW
    dow_entropy,                       # NEW
    type_entropy,                      # NEW
    cashin_agg.to_pandas(),            # NEW
    cashout_agg.to_pandas(),           # NEW
    kyc_pd[['ACCOUNT_ID', 'ACCOUNT_TYPE', 'GENDER', 'REGION', 'ACCOUNT_AGE_DAYS']],
]
for w_df in window_dfs:
    merge_list.append(w_df.to_pandas())

for df in merge_list:
    base = base.merge(df, on='ACCOUNT_ID', how='left')

print(f"  Raw feature matrix: {base.shape}")

# ============================================================
# STEP 4: DERIVED FEATURES + INTERACTIONS
# ============================================================
print("Engineering derived features...")

count_cols = [c for c in base.columns if 'COUNT' in c]
base[count_cols] = base[count_cols].fillna(0)

base['RECENCY_DAYS']  = base['RECENCY_DAYS'].fillna(91)
base['ACTIVE_DAYS']   = base['ACTIVE_DAYS'].fillna(0)
base['GAP_MAX']       = base['GAP_MAX'].fillna(91)
base['GAP_MEAN']      = base['GAP_MEAN'].fillna(91)

# Per-type recency: fill with 91 (never used that type)
for t in TRX_TYPES:
    col = f'TYPE_RECENCY_DAYS_{t}'
    if col in base.columns:
        base[col] = base[col].fillna(91)

# Core ratios
base['TRX_PER_ACTIVE_DAY']      = base['TRX_COUNT'] / (base['ACTIVE_DAYS'] + 1)
base['INCOMING_OUTGOING_RATIO'] = (base['INCOMING_COUNT'] + 1) / (base['TRX_COUNT'] + 1)
base['BAL_CV']                  = base['BAL_STD'] / (base['BAL_MEAN'].abs() + 1)
base['BAL_CHANGE']              = base['BAL_LAST'] - base['BAL_FIRST']
base['BAL_CHANGE_PCT']          = base['BAL_CHANGE'] / (base['BAL_FIRST'].abs() + 1)
base['ZERO_BAL_RATIO']          = base['ZERO_BAL_DAYS'] / (base['BAL_DAYS'] + 1)
base['LAST7_RATIO']             = (base['LAST_7D_COUNT'] + 1) / (base['TRX_COUNT'] + 1)
base['LAST14_RATIO']            = (base['LAST_14D_COUNT'] + 1) / (base['TRX_COUNT'] + 1)
base['WEEKEND_RATIO']           = base['WEEKEND_TRX_COUNT'] / (base['TRX_COUNT'] + 1)
base['NIGHT_RATIO']             = base['NIGHT_TRX_COUNT'] / (base['TRX_COUNT'] + 1)
base['AMT_IQR']                 = base['TRX_AMT_Q75'] - base['TRX_AMT_Q25']
base['MERCHANT_DIV_RATIO']      = base['UNIQUE_MERCHANTS'] / (base['UNIQUE_DST'] + 1)
base['UNIQUE_TYPE_RATIO']       = base['UNIQUE_TRX_TYPES'] / 5.0
base['BAL_LAST_VS_MEAN']        = base['BAL_LAST'] / (base['BAL_MEAN'].abs() + 1)
base['BAL_LAST7_VS_MEAN']       = base['BAL_LAST7_MEAN'] / (base['BAL_MEAN'].abs() + 1)
base['ACCOUNT_AGE_MONTHS']      = base['ACCOUNT_AGE_DAYS'] / 30.0
base['RECENCY_x_FREQ']          = base['RECENCY_DAYS'] * (1 / (base['TRX_COUNT'] + 1))
base['RECENCY_x_LAST7']         = base['RECENCY_DAYS'] * (1 / (base['LAST_7D_COUNT'] + 1))
base['ACTIVE_DAYS_x_TYPES']     = base['ACTIVE_DAYS'] * base['UNIQUE_TRX_TYPES']
base['GAP_RECENCY_RATIO']       = base['RECENCY_DAYS'] / (base['GAP_MEAN'] + 1)
base['TRX_AMT_RANGE']           = base['TRX_AMT_MAX'] - base['TRX_AMT_MIN']
base['BAL_RANGE']               = base['BAL_MAX'] - base['BAL_MIN']
base['ACTIVE_RATE']             = base['ACTIVE_DAYS'] / 91.0
base['LAST7_VS_LAST30']         = (base['LAST_7D_COUNT'] + 1) / (base['LAST_30D_COUNT'] + 1)
base['LAST14_VS_LAST60']        = (base['LAST_14D_COUNT'] + 1) / (base['LAST_60D_COUNT'] + 1)

# NEW: Cash flow features
base['CASHIN_SUM']    = base['CASHIN_SUM'].fillna(0)
base['CASHOUT_SUM']   = base['CASHOUT_SUM'].fillna(0)
base['CASHIN_COUNT']  = base['CASHIN_COUNT'].fillna(0)
base['CASHOUT_COUNT'] = base['CASHOUT_COUNT'].fillna(0)
base['NET_FLOW']      = base['CASHIN_SUM'] - base['CASHOUT_SUM']
base['FLOW_RATIO']    = (base['CASHIN_SUM'] + 1) / (base['CASHOUT_SUM'] + 1)
base['CASH_ACTIVITY_RATIO'] = (base['CASHIN_COUNT'] + base['CASHOUT_COUNT']) / (base['TRX_COUNT'] + 1)

# NEW: Trailing silence signal
base['TRAILING_INACTIVE_WEEKS'] = base['TRAILING_INACTIVE_WEEKS'].fillna(13)
base['WEEK_ACTIVITY_RATE']      = base['WEEK_ACTIVITY_RATE'].fillna(0)
base['TRAILING_ZERO_BAL_14D']   = base['TRAILING_ZERO_BAL_14D'].fillna(14)
base['TRAILING_ZERO_BAL_7D']    = base['TRAILING_ZERO_BAL_7D'].fillna(7)

# NEW: Decay score interaction
base['DECAY_SCORE']             = base['DECAY_SCORE'].fillna(0)
base['DECAY_x_RECENCY']         = base['DECAY_SCORE'] * (1 / (base['RECENCY_DAYS'] + 1))

# NEW: Week span (first to last active week)
base['WEEK_SPAN']               = base['LAST_ACTIVE_WEEK'] - base['FIRST_ACTIVE_WEEK']
base['WEEK_SPAN']               = base['WEEK_SPAN'].fillna(0)

# Log-transform skewed features
log_cols = [
    'TRX_AMT_SUM', 'TRX_AMT_MEAN', 'TRX_AMT_MAX', 'TRX_AMT_MEDIAN',
    'INCOMING_AMT', 'BAL_MEAN', 'BAL_MAX', 'BAL_LAST',
    'MERCHANT_AMT_SUM', 'LAST_7D_AMT', 'LAST_14D_AMT', 'LAST_30D_AMT',
    'TRX_COUNT', 'ACTIVE_DAYS', 'CASHIN_SUM', 'CASHOUT_SUM',
    'DECAY_AMT_SUM',
]
for col in log_cols:
    if col in base.columns:
        base[f'LOG_{col}'] = np.log1p(base[col].clip(lower=0).fillna(0))

# Safety net: any inf/-inf produced by ratio features (e.g. divisions on
# degenerate rows) gets turned into NaN so it's caught by the fillna below
# instead of being fed to the trees as +/-inf (which can also trigger
# histogram-building edge cases).
numeric_cols = base.select_dtypes(include=[np.number]).columns
base[numeric_cols] = base[numeric_cols].replace([np.inf, -np.inf], np.nan)

base = base.fillna(-1)
print(f"  Final feature matrix: {base.shape}")

# ============================================================
# STEP 5: TRAIN / TEST SPLIT + TARGET ENCODING
# ============================================================
print("Building target encoding features...")

train_base = base[base['ACCOUNT_ID'].isin(train_labels['ACCOUNT_ID'])].copy()
train_base = train_base.merge(train_labels, on='ACCOUNT_ID', how='left')
test_base  = base[base['ACCOUNT_ID'].isin(test_df['ACCOUNT_ID'])].copy()

y_train = train_base['CHURN'].values
N_FOLDS = 5
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Extended target encoding: include high-cardinality combos
TE_COLS = ['REGION', 'GENDER', 'ACCOUNT_TYPE']

# Add binned recency as TE target
train_base['RECENCY_BIN'] = pd.cut(
    train_base['RECENCY_DAYS'], bins=[-2, 1, 3, 7, 14, 30, 91, np.inf], labels=False
).fillna(-1)
test_base['RECENCY_BIN'] = pd.cut(
    test_base['RECENCY_DAYS'], bins=[-2, 1, 3, 7, 14, 30, 91, np.inf], labels=False
).fillna(-1)
TE_COLS.append('RECENCY_BIN')

for col in TE_COLS:
    if col not in train_base.columns:
        continue
    train_base[f'TE_{col}'] = 0.0
    global_mean  = y_train.mean()
    te_test_vals = np.zeros(len(test_base))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_base, y_train)):
        tr_fold  = train_base.iloc[tr_idx]
        val_fold = train_base.iloc[val_idx]
        mapping  = tr_fold.groupby(col)['CHURN'].mean()
        train_base.loc[train_base.index[val_idx], f'TE_{col}'] = (
            val_fold[col].map(mapping).fillna(global_mean).values
        )
        te_test_vals += test_base[col].map(mapping).fillna(global_mean).values / N_FOLDS

    test_base[f'TE_{col}'] = te_test_vals

FEATURE_COLS = [
    c for c in base.columns
    if c not in ['ACCOUNT_ID', 'CHURN', 'LAST_TRX_DATE']
]
FEATURE_COLS += [f'TE_{c}' for c in TE_COLS if f'TE_{c}' in train_base.columns]
FEATURE_COLS += ['RECENCY_BIN']

X_train = train_base[FEATURE_COLS].values.astype(np.float32)
X_test  = test_base[FEATURE_COLS].values.astype(np.float32)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Churn rate: {y_train.mean():.4f}")

# Label-encoded categorical columns -- tell LightGBM these are categorical
# (not ordinal/continuous) so it splits on them properly.
CAT_FEATURE_NAMES = ['ACCOUNT_TYPE', 'GENDER', 'REGION', 'RECENCY_BIN']
CAT_FEATURE_IDX = [FEATURE_COLS.index(c) for c in CAT_FEATURE_NAMES if c in FEATURE_COLS]

# ============================================================
# STEP 6: MODEL PARAMS
# ============================================================
scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())
print(f"scale_pos_weight: {scale_pos:.4f}")

lgb_params_gbdt = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'n_estimators':      6000,
    'learning_rate':     0.01,
    'num_leaves':        511,
    'max_depth':         -1,
    'min_child_samples': 20,
    'feature_fraction':  0.5,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'min_split_gain':    0.001,
    'scale_pos_weight':  scale_pos,
    'n_jobs':            -1,
    'random_state':      42,
    'verbose':           -1,
    'device':            'gpu',
    'gpu_use_dp':        True,   # double precision on GPU avoids histogram round-off crashes
}

lgb_params_gbdt2 = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'n_estimators':      6000,
    'learning_rate':     0.01,
    'num_leaves':        255,
    'max_depth':         8,
    'min_child_samples': 50,
    'feature_fraction':  0.4,
    'bagging_fraction':  0.7,
    'bagging_freq':      3,
    'reg_alpha':         0.5,
    'reg_lambda':        2.0,
    'min_split_gain':    0.001,
    'scale_pos_weight':  scale_pos,
    'n_jobs':            -1,
    'random_state':      999,
    'verbose':           -1,
    'device':            'gpu',
    'gpu_use_dp':        True,
}

lgb_params_dart = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'dart',
    'n_estimators':      2000,
    'learning_rate':     0.05,
    'num_leaves':        255,
    'max_depth':         -1,
    'min_child_samples': 30,
    'feature_fraction':  0.6,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.05,
    'reg_lambda':        0.5,
    'min_split_gain':    0.001,
    'drop_rate':         0.1,
    'scale_pos_weight':  scale_pos,
    'n_jobs':            -1,
    'random_state':      123,
    'verbose':           -1,
    'device':            'gpu',
    'gpu_use_dp':        True,
}

xgb_params = {
    'objective':             'binary:logistic',
    'eval_metric':           'auc',
    'n_estimators':          6000,
    'learning_rate':         0.01,
    'max_depth':             8,
    'min_child_weight':      20,
    'subsample':             0.8,
    'colsample_bytree':      0.5,
    'reg_alpha':             0.1,
    'reg_lambda':            1.0,
    'scale_pos_weight':      scale_pos,
    'tree_method':           'hist',
    'device':                'cuda',
    'early_stopping_rounds': 200,
    'random_state':          42,
    'verbosity':             0,
}

cat_params = {
    'iterations':            5000,
    'learning_rate':         0.02,
    'depth':                 8,
    'l2_leaf_reg':           3,
    'scale_pos_weight':      scale_pos,
    'eval_metric':           'AUC',
    'task_type':             'GPU',
    'devices':               '0',
    'random_seed':           42,
    'verbose':               500,
    'early_stopping_rounds': 200,
}

# ============================================================
# STEP 6b: SAFE LGBM FIT (auto-fallback to CPU on GPU histogram errors)
# ============================================================
def fit_lgb_safe(params, Xtr, ytr, Xval, yval, callbacks, label='', categorical_feature=None):
    """
    Fit a LightGBM model with the given params. If the GPU build hits the
    known 'Check failed: (best_split_info.left_count) > (0)' histogram
    precision bug (or any other LightGBMError), retry once on CPU so the
    whole pipeline doesn't crash mid-run.
    """
    cat_feat = categorical_feature if categorical_feature else 'auto'
    try:
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=callbacks,
              categorical_feature=cat_feat)
        return m
    except lgb.basic.LightGBMError as e:
        print(f"  [WARN] {label} GPU training failed ({e}); retrying on CPU...")
        cpu_params = dict(params)
        cpu_params['device'] = 'cpu'
        cpu_params.pop('gpu_use_dp', None)
        m = lgb.LGBMClassifier(**cpu_params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=callbacks,
              categorical_feature=cat_feat)
        return m

# ============================================================
# STEP 7: 5-FOLD CV TRAINING (5 models)
# ============================================================
oof_lgb1 = np.zeros(len(X_train))
oof_lgb2 = np.zeros(len(X_train))
oof_lgb3 = np.zeros(len(X_train))
oof_xgb  = np.zeros(len(X_train))
oof_cat  = np.zeros(len(X_train))

pred_lgb1 = np.zeros(len(X_test))
pred_lgb2 = np.zeros(len(X_test))
pred_lgb3 = np.zeros(len(X_test))
pred_xgb  = np.zeros(len(X_test))
pred_cat  = np.zeros(len(X_test))

print("\nTraining 5-model ensemble with 5-fold CV...")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*60}")
    Xtr,  Xval = X_train[tr_idx], X_train[val_idx]
    ytr,  yval = y_train[tr_idx], y_train[val_idx]

    # ---- LightGBM GBDT (deep) ----
    m1 = fit_lgb_safe(
        lgb_params_gbdt, Xtr, ytr, Xval, yval,
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)],
        label='LGB-GBDT-Deep', categorical_feature=CAT_FEATURE_IDX
    )
    oof_lgb1[val_idx]  = m1.predict_proba(Xval)[:, 1]
    pred_lgb1         += m1.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  LGB-GBDT-Deep  AUC: {roc_auc_score(yval, oof_lgb1[val_idx]):.5f}")
    del m1; gc.collect()

    # ---- LightGBM GBDT (regularized) ----
    m2 = fit_lgb_safe(
        lgb_params_gbdt2, Xtr, ytr, Xval, yval,
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)],
        label='LGB-GBDT-Reg', categorical_feature=CAT_FEATURE_IDX
    )
    oof_lgb2[val_idx]  = m2.predict_proba(Xval)[:, 1]
    pred_lgb2         += m2.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  LGB-GBDT-Reg   AUC: {roc_auc_score(yval, oof_lgb2[val_idx]):.5f}")
    del m2; gc.collect()

    # ---- LightGBM DART ----
    m3 = fit_lgb_safe(
        lgb_params_dart, Xtr, ytr, Xval, yval,
        callbacks=[lgb.log_evaluation(1000)],
        label='LGB-DART', categorical_feature=CAT_FEATURE_IDX
    )
    oof_lgb3[val_idx]  = m3.predict_proba(Xval)[:, 1]
    pred_lgb3         += m3.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  LGB-DART       AUC: {roc_auc_score(yval, oof_lgb3[val_idx]):.5f}")
    del m3; gc.collect()

    # ---- XGBoost ----
    m4 = xgb.XGBClassifier(**xgb_params)
    m4.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=1000)
    oof_xgb[val_idx]  = m4.predict_proba(Xval)[:, 1]
    pred_xgb         += m4.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  XGB            AUC: {roc_auc_score(yval, oof_xgb[val_idx]):.5f}")
    del m4; gc.collect()

    # ---- CatBoost ----
    m5 = CatBoostClassifier(**cat_params)
    m5.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    oof_cat[val_idx]  = m5.predict_proba(Xval)[:, 1]
    pred_cat         += m5.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  CatBoost       AUC: {roc_auc_score(yval, oof_cat[val_idx]):.5f}")
    del m5; gc.collect()

print(f"\n{'='*60}")
print(f"OOF LGB-GBDT-Deep : {roc_auc_score(y_train, oof_lgb1):.5f}")
print(f"OOF LGB-GBDT-Reg  : {roc_auc_score(y_train, oof_lgb2):.5f}")
print(f"OOF LGB-DART      : {roc_auc_score(y_train, oof_lgb3):.5f}")
print(f"OOF XGB           : {roc_auc_score(y_train, oof_xgb):.5f}")
print(f"OOF CatBoost      : {roc_auc_score(y_train, oof_cat):.5f}")

# ============================================================
# STEP 8: RANK-BASED OPTIMAL BLEND (5 models)
# ============================================================
print("\nOptimizing blend weights via rank averaging...")

def rank_norm(arr):
    return rankdata(arr) / len(arr)

oof_r1 = rank_norm(oof_lgb1)
oof_r2 = rank_norm(oof_lgb2)
oof_r3 = rank_norm(oof_lgb3)
oof_r4 = rank_norm(oof_xgb)
oof_r5 = rank_norm(oof_cat)

pred_r1 = rank_norm(pred_lgb1)
pred_r2 = rank_norm(pred_lgb2)
pred_r3 = rank_norm(pred_lgb3)
pred_r4 = rank_norm(pred_xgb)
pred_r5 = rank_norm(pred_cat)

# Grid search over 5-model weights
best_auc, best_w = 0.0, (0.2, 0.2, 0.2, 0.2, 0.2)
for w1 in np.arange(0.10, 0.50, 0.05):
    for w2 in np.arange(0.05, 0.40, 0.05):
        for w3 in np.arange(0.05, 0.35, 0.05):
            for w4 in np.arange(0.05, 0.35, 0.05):
                w5 = 1.0 - w1 - w2 - w3 - w4
                if w5 < 0.05 or w5 > 0.50:
                    continue
                blended = w1*oof_r1 + w2*oof_r2 + w3*oof_r3 + w4*oof_r4 + w5*oof_r5
                auc = roc_auc_score(y_train, blended)
                if auc > best_auc:
                    best_auc, best_w = auc, (w1, w2, w3, w4, w5)

w1, w2, w3, w4, w5 = best_w
print(f"Best weights -> LGB1:{w1:.2f} LGB2:{w2:.2f} LGB3:{w3:.2f} XGB:{w4:.2f} CAT:{w5:.2f}")
print(f"Best OOF AUC (rank blend): {best_auc:.5f}")

final_pred = w1*pred_r1 + w2*pred_r2 + w3*pred_r3 + w4*pred_r4 + w5*pred_r5

# ============================================================
# STEP 9: SUBMISSION
# ============================================================
pred_map   = dict(zip(test_base['ACCOUNT_ID'].values, final_pred))
submission = test_df[['ACCOUNT_ID']].copy()
submission['CHURN_PROB'] = submission['ACCOUNT_ID'].map(pred_map)
submission['CHURN_PROB'] = submission['CHURN_PROB'].fillna(submission['CHURN_PROB'].median())

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission['CHURN_PROB'].describe())
print(submission.head(10))
