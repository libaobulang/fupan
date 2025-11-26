import os
import re
import argparse
from functools import reduce
from datetime import datetime
import pandas as pd
import pywencai

BOARD_CSV = os.path.join(os.path.dirname(__file__), 'data', '板块资金与涨幅.csv')
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)

def _pick_df(r):
    if isinstance(r, pd.DataFrame) and not r.empty:
        return r
    if isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
        return r[0]
    if isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
        return r['tableV1']
    return pd.DataFrame()

def _parse_amt(val):
    if pd.api.types.is_number(val):
        return float(val)
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

def fetch_and_write(date: str) -> bool:
    try:
        q = "同花顺概念指数 成交额,上1交易日的成交额,板块资金净额,涨跌幅,涨停家数"
        r = pywencai.get(query=q, query_type='zhishu', loop=True)
        df = _pick_df(r)
        if df.empty:
            q2 = "同花顺概念指数 成交额(元),上1交易日的成交额(元),资金流向(元),涨停家数,涨跌幅(%)"
            df = _pick_df(pywencai.get(query=q2, query_type='zhishu', loop=True))
        if df.empty:
            return False
        # 清洗列名（去掉日期标签）
        tag = date
        try:
            d_dot = datetime.strptime(tag, '%Y%m%d').strftime('%Y.%m.%d')
            d_dash = datetime.strptime(tag, '%Y%m%d').strftime('%Y-%m-%d')
            prev = (datetime.strptime(tag, '%Y%m%d') - pd.Timedelta(days=1))
            prev_dot = prev.strftime('%Y.%m.%d')
            prev_dash = prev.strftime('%Y-%m-%d')
        except Exception:
            d_dot = tag
            d_dash = tag
            prev_dot = tag
            prev_dash = tag
        df = df.copy()
        df.columns = [re.sub(r"\[\d{8}\]", "", str(c)).strip() for c in df.columns]
        df.columns = [reduce(lambda s,t: s.replace(t, ''), [d_dot, d_dash, prev_dot, prev_dash], str(c)).strip() for c in df.columns]
        cols = [str(c) for c in df.columns]
        def find(tokens):
            for c in cols:
                cc = str(c)
                if all(t in cc for t in tokens):
                    return c
            return None
        code_col = find(['指数代码']) or find(['code']) or find(['market_code'])
        name_col = find(['指数简称']) or find(['指数名称']) or find(['指数'])
        amt_today_col = find(['成交额'])
        prev_amt_col = find(['上1','成交额']) or find(['上一','成交额'])
        inflow_col = find(['板块资金净额']) or find(['资金流向']) or find(['资金流向(元)']) or find(['资金净额'])
        pct_col = find(['涨跌幅'])
        zt_col = find(['涨停家数'])
        # 过滤概念指数
        filt = df
        if code_col is not None:
            try:
                filt = df[df[code_col].astype(str).str.contains('\.TI')]
            except Exception:
                filt = df
        rows = []
        for _, r0 in filt.iterrows():
            code = str(r0[code_col]) if code_col else ''
            name = str(r0[name_col]) if name_col else ''
            amt_today = _parse_amt(r0[amt_today_col]) if amt_today_col else pd.NA
            prev_amt = _parse_amt(r0[prev_amt_col]) if prev_amt_col else pd.NA
            inflow = _parse_amt(r0[inflow_col]) if inflow_col else pd.NA
            pct = pd.to_numeric(str(r0[pct_col]).replace('%',''), errors='coerce') if pct_col else pd.NA
            zt = pd.to_numeric(str(r0[zt_col]), errors='coerce') if zt_col else pd.NA
            rows.append({
                '日期': tag,
                '指数代码': code,
                '指数名称': name,
                '成交额(元)': amt_today,
                '上1交易日成交额(元)': prev_amt,
                '资金流向(元)': inflow,
                '涨停家数(家)': zt,
                '涨跌幅(%)': pct,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return False
        # 合并到板块资金与涨幅.csv
        base = pd.read_csv(BOARD_CSV, encoding='utf-8-sig') if os.path.exists(BOARD_CSV) else pd.DataFrame()
        if not base.empty:
            base = base[base['日期'].astype(str) != str(tag)]
            out = pd.concat([base, out], ignore_index=True)
        out = out[['日期','指数代码','指数名称','成交额(元)','上1交易日成交额(元)','资金流向(元)','涨停家数(家)','涨跌幅(%)']]
        if '指数代码' in out.columns:
            out = out.drop_duplicates(subset=['日期','指数代码'], keep='last')
        out.to_csv(BOARD_CSV, index=False, encoding='utf-8-sig')
        return True
    except Exception:
        return False

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--date')
    args = ap.parse_args()
    d = args.date if args.date else datetime.now().strftime('%Y%m%d')
    ok = fetch_and_write(d)
    print('OK' if ok else 'FAIL')