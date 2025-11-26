import os
from datetime import datetime
import argparse
import pandas as pd
import pywencai
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def out_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

def pick_df(r):
    if isinstance(r, pd.DataFrame) and not r.empty:
        return r
    if isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
        return r[0]
    if isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
        return r['tableV1']
    return pd.DataFrame()

def main(date: str | None):
    tag = date if date else datetime.now().strftime('%Y%m%d')
    queryboard = f"{tag} 同花顺概念指数的 当日成交额,较上一交易日成交额,主力净流入,近三日大单净额"
    r = pywencai.get(query=queryboard, query_type='zhishu', loop=True)
    df = pick_df(r)
    p = out_path(f'probe_zhishu_{tag}.csv')
    df.to_csv(p, index=False, encoding='utf-8-sig')
    print(p)
    try:
        d = tag
        d_dot = datetime.strptime(d, '%Y%m%d').strftime('%Y.%m.%d')
        d_dash = datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
    except Exception:
        d_dot = d
        d_dash = d
    cols = [str(c).strip() for c in df.columns]
    def find_col(tokens):
        for c in cols:
            n = str(c)
            if all(t in n for t in tokens):
                return c
        return None
    code_col = find_col(['指数代码']) or find_col(['code']) or find_col(['market_code'])
    name_col = find_col(['指数简称']) or find_col(['指数名称'])
    amt_today_col = find_col(['指数@成交额', d]) or find_col(['指数@成交额', d_dot]) or find_col(['指数@成交额', d_dash]) or find_col(['当日成交额']) or find_col(['成交额(亿)'])
    try:
        prev = datetime.strptime(d, '%Y%m%d') - pd.Timedelta(days=1)
        prev_d = prev.strftime('%Y%m%d')
        prev_dot = prev.strftime('%Y.%m.%d')
        prev_dash = prev.strftime('%Y-%m-%d')
    except Exception:
        prev_d = None
        prev_dot = None
        prev_dash = None
    amt_prev_col = (find_col(['指数@成交额', prev_d]) or find_col(['指数@成交额', prev_dot]) or find_col(['指数@成交额', prev_dash]))
    pct_col = find_col(['指数@涨跌幅', '前复权', d]) or find_col(['指数@涨跌幅', '前复权', d_dot]) or find_col(['指数@涨跌幅', '前复权', d_dash])
    inflow_col = find_col(['主力资金流向']) or find_col(['主力净流入']) or find_col(['主力资金净流入'])
    big3_col = find_col(['区间dde大单净额']) or find_col(['近三', '大单净额'])
    def parse_amt_unit(val):
        s = str(val)
        mul = 1.0
        if ('亿' in s) or ('亿元' in s):
            mul = 1e8
        elif ('万' in s) or ('万元' in s):
            mul = 1e4
        m = re.findall(r"[-+]?\d[\d,]*\.?\d*", s)
        ts = m[0] if m else ''
        v = pd.to_numeric(str(ts).replace(',', ''), errors='coerce')
        return v*mul if pd.notna(v) else pd.NA
    rows = []
    filt = df
    if code_col is not None:
        try:
            filt = df[df[code_col].astype(str).str.contains('\.TI|\.CSI|\.SZ|\.SH')]
        except Exception:
            filt = df
    for _, r in filt.iterrows():
        code = str(r[code_col]) if code_col is not None else ''
        name = str(r[name_col]) if name_col is not None else ''
        amt_today = parse_amt_unit(r[amt_today_col]) if amt_today_col is not None else pd.NA
        amt_prev = parse_amt_unit(r[amt_prev_col]) if amt_prev_col is not None else pd.NA
        pct = pd.to_numeric(str(r[pct_col]).replace('%',''), errors='coerce') if pct_col is not None else pd.NA
        inflow = parse_amt_unit(r[inflow_col]) if inflow_col is not None else pd.NA
        big3 = parse_amt_unit(r[big3_col]) if big3_col is not None else pd.NA
        delta = (amt_today - amt_prev) if (pd.notna(amt_today) and pd.notna(amt_prev)) else pd.NA
        rows.append({
            '日期': d,
            '指数代码': code,
            '指数名称': name,
            '成交额(元)': amt_today,
            '涨跌幅(%)': pct,
            '主力净流入(元)': inflow,
            '较上一交易日成交额(元)': delta,
            '近三交易日成交额(元)': pd.NA,
            '近三日大单净额(元)': big3,
        })
    out = pd.DataFrame(rows)
    board_path = out_path('板块资金与涨幅.csv')
    if os.path.exists(board_path):
        base = pd.read_csv(board_path, encoding='utf-8-sig')
        base = base[base['日期'].astype(str) != str(d)]
        out = pd.concat([base, out], ignore_index=True)
    out = out[['日期','指数代码','指数名称','成交额(元)','涨跌幅(%)','主力净流入(元)','较上一交易日成交额(元)','近三交易日成交额(元)','近三日大单净额(元)']]
    if '指数代码' in out.columns:
        out = out.drop_duplicates(subset=['日期','指数代码'], keep='last')
    out.to_csv(board_path, index=False, encoding='utf-8-sig')
    print(board_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date')
    args = parser.parse_args()
    main(args.date)