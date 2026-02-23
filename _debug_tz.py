import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from market_close_times import MARKET_CLOSE_TIMES

def _currency_to_timezone(currency):
    info = MARKET_CLOSE_TIMES.get(str(currency).upper())
    return info['timezone'] if info else None

def _parse_exec_time_to_utc(val):
    if pd.isna(val): return None
    raw = str(val).split('[',1)[0]
    parsed = pd.to_datetime(raw, errors='coerce', utc=True)
    if pd.isna(parsed): return None
    return parsed.to_pydatetime()

def _to_local(utc_dt, tz):
    if utc_dt is None: return None
    if not tz: return utc_dt
    return utc_dt.astimezone(ZoneInfo(tz))

df = pd.read_csv('output/synthetic_trading_data.csv')

position_keys = ['portfolioId', 'underlyingId', 'maturity', 'underlyingCurrency']
df['market_timezone'] = df['underlyingCurrency'].apply(_currency_to_timezone)
df['execTime_parsed_utc'] = df['execTime'].apply(_parse_exec_time_to_utc)
df['execTime_parsed'] = df.apply(lambda r: _to_local(r['execTime_parsed_utc'], r['market_timezone']), axis=1)
df['hour_bucket'] = df['execTime_parsed'].apply(lambda ts: ts.replace(minute=0, second=0, microsecond=0, tzinfo=None) if ts else None)
df['signed_qty'] = np.where(df['way'].str.upper() == 'BUY', df['quantity'], -df['quantity'])

# Check BEFORE groupby
chf_mask = (df['underlyingCurrency']=='CHF') & (df['underlyingId'].astype(str)=='13440') & (df['portfolioId']=='CTM_EQUITY')
chf_pre = df[chf_mask & df['execTime'].str.startswith('2024-02-14')]
print('=== BEFORE GROUPBY (CHF/13440/CTM_EQUITY/2024-02-14) ===')
print(f'hour_bucket column dtype: {df["hour_bucket"].dtype}')
sample = chf_pre['hour_bucket'].sort_values().head(3)
for v in sample:
    print(f'  {v}  tzinfo={v.tzinfo}  offset={v.utcoffset()}')

# Now do the groupby
hourly_net = (
    df.groupby(position_keys + ['hour_bucket'])['signed_qty']
    .sum()
    .reset_index()
    .sort_values(by=position_keys + ['hour_bucket'])
)

print(f'\n=== AFTER GROUPBY ===')
print(f'hour_bucket column dtype: {hourly_net["hour_bucket"].dtype}')

chf_post = hourly_net[
    (hourly_net['underlyingCurrency']=='CHF') &
    (hourly_net['underlyingId'].astype(str)=='13440') &
    (hourly_net['portfolioId']=='CTM_EQUITY')
].copy()
chf_post['execDate'] = chf_post['hour_bucket'].apply(lambda ts: ts.date() if hasattr(ts, 'date') else None)
chf_feb14 = chf_post[chf_post['execDate'] == pd.Timestamp('2024-02-14').date()]
print(f'Rows for 2024-02-14: {len(chf_feb14)}')
print('Buckets after groupby:')
for _, r in chf_feb14.iterrows():
    b = r['hour_bucket']
    tz = getattr(b, 'tzinfo', 'N/A')
    off = b.utcoffset() if hasattr(b, 'utcoffset') else 'N/A'
    print(f'  {b}  type={type(b).__name__}  tzinfo={tz}  offset={off}')
