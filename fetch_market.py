import pandas as pd
import pywencai
import numpy as np
import re
import os
import argparse
from datetime import datetime

pattern = r"\[\d{8}\]"

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache_market')
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(date):
    return os.path.join(CACHE_DIR, f'市场情绪_{date}.csv')

def enhanced_path(date):
    return os.path.join(os.path.dirname(BASE_DIR), f'增强涨停{date}.csv')

def clean_columns(df):
    df.columns = [re.sub(pattern, "", str(c)).strip() for c in df.columns]
    return df

def to_num(v):
    return pd.to_numeric(str(v).replace('%','').replace(',',''), errors='coerce')

def extract_indices(df):
    df = clean_columns(df)
    name_cols = [c for c in df.columns if any(s in str(c) for s in ['指数简称','指数名称','指数','名称','简称'])]
    pct_cols = [c for c in df.columns if '涨跌幅' in str(c)]
    res = {'上证指数涨跌幅': np.nan, '深证成指涨跌幅': np.nan, '创业板指涨跌幅': np.nan}
    def pick(name):
        for nc in name_cols:
            m = df[df[nc].astype(str).str.contains(name, na=False)]
            if not m.empty:
                for pc in pct_cols:
                    return to_num(m[pc].iloc[0])
        return np.nan
    res['上证指数涨跌幅'] = pick('上证指数')
    res['深证成指涨跌幅'] = pick('深证成指')
    res['创业板指涨跌幅'] = pick('创业板指')
    return res

def fetch_indices(date):
    q = f'{date}上证指数涨跌幅,深证成指涨跌幅,创业板指涨跌幅'
    try:
        r = pywencai.get(query=q, loop=True)
    except Exception:
        r = None
    if isinstance(r, pd.DataFrame) and not r.empty:
        return extract_indices(r)
    if isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame) and not r[0].empty:
        return extract_indices(r[0])
    def single(name):
        try:
            rr = pywencai.get(query=f'{date}{name}涨跌幅', loop=True)
            if isinstance(rr, pd.DataFrame) and not rr.empty:
                rr = clean_columns(rr)
                pc = next((c for c in rr.columns if '涨跌幅' in str(c)), None)
                if pc:
                    return to_num(rr[pc].iloc[0])
            d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            rr2 = pywencai.get(query=f'{d_dash}{name}涨跌幅', loop=True)
            if isinstance(rr2, pd.DataFrame) and not rr2.empty:
                rr2 = clean_columns(rr2)
                pc2 = next((c for c in rr2.columns if '涨跌幅' in str(c)), None)
                if pc2:
                    return to_num(rr2[pc2].iloc[0])
        except Exception:
            pass
        return np.nan
    return {
        '上证指数涨跌幅': single('上证指数'),
        '深证成指涨跌幅': single('深证成指'),
        '创业板指涨跌幅': single('创业板指'),
    }

def fetch_breadth(date):
    q = f'{date}涨停家数,跌停家数,上涨家数,下跌家数'
    up = down = dt = zt = np.nan
    try:
        try:
            r = pywencai.get(query=q, loop=True)
        except Exception:
            r = None
        if isinstance(r, pd.DataFrame) and not r.empty:
            r = clean_columns(r)
            for c in r.columns:
                n = str(c)
                if '上涨家数' in n:
                    up = to_num(r[c].iloc[0])
                if '下跌家数' in n:
                    down = to_num(r[c].iloc[0])
                if '跌停家数' in n:
                    dt = to_num(r[c].iloc[0])
                if '涨停家数' in n:
                    zt = to_num(r[c].iloc[0])
    except Exception:
        pass
    if pd.isna(dt):
        try:
            rr = pywencai.get(query=f'{date}跌停股票,非st股票', loop=True)
            if isinstance(rr, pd.DataFrame):
                dt = len(rr)
            elif isinstance(rr, list) and len(rr) > 0 and isinstance(rr[0], pd.DataFrame):
                dt = len(rr[0])
        except Exception:
            pass
    if pd.isna(up) or pd.isna(down):
        try:
            rr = pywencai.get(query=f'{date}A股股票,最新涨跌幅,非st股票', loop=True)
            df = rr if isinstance(rr, pd.DataFrame) else (rr[0] if isinstance(rr, list) and len(rr) > 0 and isinstance(rr[0], pd.DataFrame) else None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = clean_columns(df)
                pc = next((c for c in df.columns if '涨跌幅' in str(c)), None)
                if pc:
                    s = pd.to_numeric(df[pc].astype(str).str.replace('%',''), errors='coerce')
                    up = int((s > 0).sum())
                    down = int((s < 0).sum())
        except Exception:
            pass
    return {'上涨家数': up, '下跌家数': down, '跌停家数': dt, '涨停家数': zt}

def merge_cache(date, data):
    p = cache_path(date)
    base = None
    if os.path.exists(p):
        try:
            base = pd.read_csv(p, encoding='utf-8-sig')
        except Exception:
            base = None
    if isinstance(base, pd.DataFrame) and not base.empty:
        row = base.iloc[0].to_dict()
        for k, v in data.items():
            if not (pd.isna(v)):
                row[k] = v
        out = pd.DataFrame([row])
    else:
        out = pd.DataFrame([data])
    out.to_csv(p, index=False, encoding='utf-8-sig')
    return out

def backfill_stocks(date):
    p = cache_path(date)
    try:
        m = pd.read_csv(p, encoding='utf-8-sig')
    except Exception:
        m = None
    if not (isinstance(m, pd.DataFrame) and not m.empty):
        print('缓存不存在或为空')
        return False
    row = m.iloc[0].to_dict()
    zt = pd.to_numeric(row.get('涨停家数'), errors='coerce')
    dt = pd.to_numeric(row.get('跌停家数'), errors='coerce')
    up = pd.to_numeric(row.get('上涨家数'), errors='coerce')
    down = pd.to_numeric(row.get('下跌家数'), errors='coerce')
    total = (up + down) if pd.notna(up) and pd.notna(down) else np.nan
    up_ratio = (up / (total + 1e-5)) if pd.notna(total) else np.nan
    sentiment = np.nan
    if pd.notna(up_ratio) and pd.notna(zt) and pd.notna(dt):
        sentiment = (up_ratio * 0.4 + min(max(zt / 100, 0), 1) * 0.3 + (1 - min(max(dt / 50, 0), 1)) * 0.3) * 100
    ep = enhanced_path(date)
    try:
        df = pd.read_csv(ep, encoding='utf-8-sig')
    except Exception as e:
        print(f'读取增强文件失败: {e}')
        return False
    df['市场涨停家数'] = zt
    df['市场跌停家数'] = dt
    if pd.notna(sentiment):
        df['市场情绪分数'] = sentiment
    df.to_csv(ep, index=False, encoding='utf-8-sig')
    print(df[['市场涨停家数','市场跌停家数','市场情绪分数']].head(3).to_string(index=False))
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date')
    ap.add_argument('--indices', action='store_true')
    ap.add_argument('--breadth', action='store_true')
    ap.add_argument('--backfill-stocks', action='store_true')
    args = ap.parse_args()
    d = args.date or datetime.now().strftime('%Y%m%d')
    if args.indices:
        idx = fetch_indices(d)
        out = merge_cache(d, idx)
        print(out.to_string(index=False))
    if args.breadth:
        br = fetch_breadth(d)
        out = merge_cache(d, br)
        print(out.to_string(index=False))
    if args.backfill_stocks:
        backfill_stocks(d)
    if not args.indices and not args.breadth:
        idx = fetch_indices(d)
        br = fetch_breadth(d)
        data = {}
        data.update(idx)
        data.update(br)
        out = merge_cache(d, data)
        print(out.to_string(index=False))

if __name__ == '__main__':
    main()