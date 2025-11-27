import pandas as pd
import pywencai
import akshare as ak
import re
from functools import reduce
import numpy as np
from collections import Counter
from datetime import datetime
import time
import os
import shutil
import argparse
import baostock as bs

pattern = r"\[\d{8}\]"
all_dfs = []  # 用于存储所有 DataFrame 的列表

# 默认处理当天，确保脚本可直接运行
datelist = [datetime.now().strftime('%Y%m%d')]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
CACHE_DIR = os.path.join(DATA_DIR, 'cache_market')
os.makedirs(CACHE_DIR, exist_ok=True)

def market_cache_path(date):
    return os.path.join(CACHE_DIR, f'市场情绪_{date}.csv')

def all_market_cache_path():
    return os.path.join(CACHE_DIR, '市场情绪.csv')

BLACKLIST_CONCEPTS = set([
    '融资融券','沪股通','深股通','证金持股','同花顺新质50','昨日成交前十','同花顺拟回购指数','百元股','中小100','高股息精选','高市盈率','高市净率','深股通成交前十','机构周调研前十'
])

def load_concept_config():
    base = {
        'blacklist': list(BLACKLIST_CONCEPTS),
        'pct_div': 5.0,
        'amt_scale': 1e9,
        'amt_div': 50.0,
        'hot_score_min': 3.0,
        'hot_topn': 20,
        'board_score_weights': [0.5, 0.3, 0.2]
    }
    p = os.path.join(os.path.dirname(__file__), 'concept_config.json')
    try:
        if os.path.exists(p):
            import json
            with open(p, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            for k, v in base.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    except Exception:
        pass
    return base

CONCEPT_CFG = load_concept_config()
BLACKLIST_CONCEPTS = set(CONCEPT_CFG.get('blacklist', list(BLACKLIST_CONCEPTS)))

def md_table(headers, rows):
    line1 = '| ' + ' | '.join(headers) + ' |'
    line2 = '| ' + ' | '.join(['---']*len(headers)) + ' |'
    lines = [line1, line2]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)

def fetch_total_amt(date):
    def _pick_df(r):
        if isinstance(r, pd.DataFrame) and not r.empty:
            return r
        if isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
            return r[0]
        if isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
            return r['tableV1']
        return None
    def _parse_amt_str(v, col_name):
        try:
            s = str(v)
            unit = 1.0
            if '亿' in s:
                unit = 1e8
            elif '万' in s:
                unit = 1e4
            s_clean = re.sub(r"[^0-9\.\-]", "", s)
            x = pd.to_numeric(s_clean, errors='coerce')
            if pd.notna(x):
                return float(x) * unit
            # 列名兜底单位
            x2 = pd.to_numeric(str(v).replace(',',''), errors='coerce')
            if pd.notna(x2):
                if ('(亿)' in str(col_name)) or ('亿元' in str(col_name)):
                    return float(x2) * 1e8
                return float(x2)
        except Exception:
            pass
        return np.nan
    # 方案1：直接取全A的成交额（带日期标签与不带日期两种）
    for q in [
        f"{date}同花顺全A(沪深京)的成交额",
        "同花顺全A(沪深京)的成交额",
        f"{date}同花顺全A(沪深京)的成交额(亿)",
        "同花顺全A(沪深京)的成交额(亿)",
    ]:
        try:
            r = pywencai.get(query=q, query_type='zhishu', loop=True)
        except Exception:
            r = None
        df0 = _pick_df(r)
        if isinstance(df0, pd.DataFrame) and not df0.empty:
            df0 = df0.copy()
            df0.columns = [re.sub(pattern, "", str(c)).strip() for c in df0.columns]
            cols = list(df0.columns)
            name_col = next((c for c in cols if ('指数' in str(c)) and (('简称' in str(c)) or ('名称' in str(c)))), None)
            code_col = next((c for c in cols if ('指数代码' in str(c))), None)
            amt_cols = [c for c in cols if ('成交额' in str(c))]
            sub = df0
            if code_col is not None:
                sub = df0[df0[code_col].astype(str) == '883957.TI']
                if sub.empty and name_col is not None:
                    sub = df0[df0[name_col].astype(str).str.contains('同花顺全A', na=False)]
            elif name_col is not None:
                sub = df0[df0[name_col].astype(str).str.contains('同花顺全A', na=False)]
            if not sub.empty and len(amt_cols) > 0:
                val = _parse_amt_str(sub[amt_cols[0]].iloc[0], amt_cols[0])
                if pd.notna(val):
                    return float(val)
    # 方案2：沪/深/北三市成交额相加（元或亿）
    try:
        q2 = f"{date}沪市成交额,深市成交额,北证成交额"
        r2 = pywencai.get(query=q2, query_type='zhishu', loop=True)
        df2 = _pick_df(r2)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            df2 = df2.copy()
            df2.columns = [re.sub(pattern, "", str(c)).strip() for c in df2.columns]
            total = 0.0
            for tok in ['沪市成交额', '深市成交额', '北证成交额']:
                col = next((c for c in df2.columns if tok in str(c)), None)
                if col:
                    v = _parse_amt_str(df2[col].iloc[0], col)
                    total += float(v) if pd.notna(v) else 0.0
            return total if total > 0 else np.nan
    except Exception:
        pass
    try:
        q3 = f"{date}沪市成交金额,深市成交金额,北证成交金额"
        r3 = pywencai.get(query=q3, query_type='zhishu', loop=True)
        df3 = _pick_df(r3)
        if isinstance(df3, pd.DataFrame) and not df3.empty:
            df3 = df3.copy()
            df3.columns = [re.sub(pattern, "", str(c)).strip() for c in df3.columns]
            total = 0.0
            for tok in ['沪市成交金额', '深市成交金额', '北证成交金额']:
                col = next((c for c in df3.columns if tok in str(c)), None)
                if col:
                    v = _parse_amt_str(df3[col].iloc[0], col)
                    total += float(v) if pd.notna(v) else 0.0
            return total if total > 0 else np.nan
    except Exception:
        pass
    try:
        q4 = f"{date}沪市成交额(亿),深市成交额(亿),北证成交额(亿)"
        r4 = pywencai.get(query=q4, query_type='zhishu', loop=True)
        df4 = _pick_df(r4)
        if isinstance(df4, pd.DataFrame) and not df4.empty:
            df4 = df4.copy()
            df4.columns = [re.sub(pattern, "", str(c)).strip() for c in df4.columns]
            total = 0.0
            for tok in ['沪市成交额(亿)', '深市成交额(亿)', '北证成交额(亿)']:
                col = next((c for c in df4.columns if tok in str(c)), None)
                if col:
                    v = _parse_amt_str(df4[col].iloc[0], col)
                    total += float(v) if pd.notna(v) else 0.0
            return total if total > 0 else np.nan
    except Exception:
        pass
    try:
        q5 = f"{date}市场情绪,成交额,上涨家数,下跌家数,平盘家数"
        r5 = pywencai.get(query=q5, query_type='zhishu', loop=True)
        df5 = _pick_df(r5)
        if isinstance(df5, pd.DataFrame) and not df5.empty:
            df5 = df5.copy()
            df5.columns = [re.sub(pattern, "", str(c)).strip() for c in df5.columns]
            cols = list(df5.columns)
            code_col = next((c for c in cols if ('指数代码' in str(c))), None)
            name_col = next((c for c in cols if ('指数' in str(c)) and (('简称' in str(c)) or ('名称' in str(c)))), None)
            amt_cols = [c for c in cols if ('成交额' in str(c))]
            sub = df5
            if code_col is not None:
                sub = df5[df5[code_col].astype(str) == '883957.TI']
                if sub.empty and name_col is not None:
                    sub = df5[df5[name_col].astype(str).str.contains('同花顺全A', na=False)]
            elif name_col is not None:
                sub = df5[df5[name_col].astype(str).str.contains('同花顺全A', na=False)]
            if not sub.empty and len(amt_cols) > 0:
                v = _parse_amt_str(sub[amt_cols[0]].iloc[0], amt_cols[0])
                if pd.notna(v):
                    return float(v)
    except Exception:
        pass
    # 不再从板块文件汇总概念成交额（会重复叠加导致明显偏大）
    try:
        spot = ak.stock_zh_a_spot_em()
        if isinstance(spot, pd.DataFrame) and not spot.empty:
            # 兼容不同列名：成交额/成交金额/成交额(元)
            cols = [c for c in spot.columns if '成交' in str(c) and ('额' in str(c) or '金额' in str(c))]
            for col in cols:
                s = pd.to_numeric(spot[col].astype(str).str.replace(',',''), errors='coerce')
                total = float(s.sum()) if pd.notna(s.sum()) else np.nan
                if pd.notna(total) and total > 0:
                    return total
    except Exception:
        pass
    return np.nan

def rebuild_all_cache_from_files():
    try:
        rows = []
        for name in os.listdir(CACHE_DIR):
            if name.startswith('市场情绪_') and name.endswith('.csv'):
                p = os.path.join(CACHE_DIR, name)
                try:
                    df = pd.read_csv(p, encoding='utf-8-sig')
                    date = name.replace('市场情绪_', '').replace('.csv', '')
                    df['日期'] = date
                    rows.append(df)
                except Exception:
                    continue
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            # 标准化列
            cols = ['日期','上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅','上涨家数','下跌家数','平盘家数','涨停家数','跌停家数']
            for c in cols:
                if c not in all_df.columns:
                    all_df[c] = np.nan
            all_df = all_df[cols]
            # 去重保留最新
            all_df = all_df.drop_duplicates(subset=['日期'], keep='last')
            all_df = all_df.sort_values('日期')
            all_df.to_csv(all_market_cache_path(), index=False, encoding='utf-8-sig')
            return True
    except Exception:
        pass
    return False

def load_market_cache(date):
    p_all = all_market_cache_path()
    if os.path.exists(p_all):
        try:
            df = pd.read_csv(p_all, encoding='utf-8-sig')
            if '日期' in df.columns:
                row = df[df['日期'].astype(str) == str(date)]
                if not row.empty:
                    return row.reset_index(drop=True)
        except Exception:
            pass
    p = market_cache_path(date)
    if os.path.exists(p):
        try:
            return pd.read_csv(p, encoding='utf-8-sig')
        except Exception:
            return None
    # 尝试从分文件重建总文件
    rebuild_all_cache_from_files()
    if os.path.exists(p_all):
        try:
            df = pd.read_csv(p_all, encoding='utf-8-sig')
            if '日期' in df.columns:
                row = df[df['日期'].astype(str) == str(date)]
                if not row.empty:
                    return row.reset_index(drop=True)
        except Exception:
            pass
    return None

def save_market_cache(date, df):
    try:
        df2 = df.copy()
        df2['日期'] = date
        cols = ['日期','上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅','上涨家数','下跌家数','平盘家数','涨停家数','跌停家数','A股总成交额(元)','情绪分数']
        for c in cols:
            if c not in df2.columns:
                df2[c] = np.nan
        try:
            up = pd.to_numeric(df2['上涨家数'], errors='coerce').iloc[0]
            down = pd.to_numeric(df2['下跌家数'], errors='coerce').iloc[0]
            zt = pd.to_numeric(df2['涨停家数'], errors='coerce').iloc[0]
            dt = pd.to_numeric(df2['跌停家数'], errors='coerce').iloc[0]
            total = (up + down) if pd.notna(up) and pd.notna(down) else np.nan
            up_ratio = (up / (total + 1e-5)) if pd.notna(total) else np.nan
            sentiment = np.nan
            if pd.notna(up_ratio) and pd.notna(zt) and pd.notna(dt):
                sentiment = (up_ratio * 0.4 + min(max(zt / 100, 0), 1) * 0.3 + (1 - min(max(dt / 50, 0), 1)) * 0.3) * 100
            df2['情绪分数'] = sentiment
        except Exception:
            pass
        try:
            v_total = pd.to_numeric(df2['A股总成交额(元)'], errors='coerce').iloc[0]
            if pd.isna(v_total):
                try:
                    v = fetch_total_amt(date)
                    if pd.notna(v):
                        df2.loc[df2.index[0], 'A股总成交额(元)'] = float(v)
                except Exception:
                    pass
        except Exception:
            pass
        df2 = df2[cols].iloc[[0]]
        p_all = all_market_cache_path()
        if os.path.exists(p_all):
            try:
                base = pd.read_csv(p_all, encoding='utf-8-sig')
            except Exception:
                base = pd.DataFrame(columns=cols)
        else:
            base = pd.DataFrame(columns=cols)
        try:
            base = base.reindex(columns=cols)
        except Exception:
            for c in cols:
                if c not in base.columns:
                    base[c] = np.nan
            base = base[cols]
        base = base[base['日期'].astype(str) != str(date)]
        out = pd.concat([base, df2], ignore_index=True)
        out.to_csv(p_all, index=False, encoding='utf-8-sig')
    except Exception:
        pass

def is_market_complete(df):
    req = ['上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅','涨停家数','跌停家数','上涨家数','下跌家数']
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    row = df.iloc[0]
    for k in req:
        if pd.isna(row.get(k)):
            return False
    return True

def extract_index_pct_from_df(df):
    d = df.copy()
    d.columns = [re.sub(pattern, "", str(c)).strip() for c in d.columns]
    idx_col = next((c for c in d.columns if '指数简称' in str(c) or '指数名称' in str(c)), None)
    pct_col = next((c for c in d.columns if '涨跌幅' in str(c)), None)
    def val(name):
        if idx_col is not None and pct_col is not None:
            sub = d[d[idx_col].astype(str) == name]
            if not sub.empty:
                v = sub[pct_col].iloc[0]
                return pd.to_numeric(str(v).replace('%',''), errors='coerce')
        return np.nan
    return {
        '上证指数涨跌幅': val('上证指数'),
        '深证成指涨跌幅': val('深证成指'),
        '创业板指涨跌幅': val('创业板指'),
    }

def normalize_columns(df):
    # 统一常用列名，避免下游 KeyError
    if '股票代码' in df.columns and 'code' not in df.columns:
        df['code'] = df['股票代码']
    if '证券代码' in df.columns and 'code' not in df.columns:
        df['code'] = df['证券代码']
    if '市盈率' in df.columns and '市盈率(pe)' not in df.columns:
        df['市盈率(pe)'] = df['市盈率']
    if '所属同花顺概念' in df.columns and '所属概念' not in df.columns:
        df['所属概念'] = df['所属同花顺概念']
    if '行业' in df.columns and '所属同花顺行业' not in df.columns:
        df['所属同花顺行业'] = df['行业']
    if '大单净量' in df.columns and '最新dde大单净额' not in df.columns:
        df['最新dde大单净额'] = df['大单净量']
    if '涨跌幅' in df.columns and '最新涨跌幅' not in df.columns:
        df['最新涨跌幅'] = df['涨跌幅']
    if '涨跌幅:前复权' in df.columns and '最新涨跌幅' not in df.columns:
        df['最新涨跌幅'] = df['涨跌幅:前复权']
    if '收盘价:不复权' in df.columns and '最新价' not in df.columns:
        df['最新价'] = df['收盘价:不复权']
    return df

def coerce_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def parse_market_cap(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    s = str(value).replace(',', '').strip()
    if s.endswith('亿'):
        try:
            return float(s[:-1]) * 1e8
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def fetch_hold_data(date, names):
    frames = []
    for name in names:
        df0 = None
        try:
            q = f"股票简称={name},最新价,涨跌幅,换手率,量比,振幅,所属同花顺概念,所属同花顺行业,行业,市盈率,最新dde大单净额,大单净量,a股市值(不含限售股)"
            r = pywencai.get(query=q, query_type='stock', loop=True)
            if isinstance(r, pd.DataFrame) and not r.empty:
                df0 = r
            elif isinstance(r, list):
                df0 = next((x for x in r if isinstance(x, pd.DataFrame) and not x.empty), None)
            elif isinstance(r, dict):
                df0 = r.get('tableV1') if isinstance(r.get('tableV1'), pd.DataFrame) else None
        except Exception:
            df0 = None
        if isinstance(df0, pd.DataFrame) and not df0.empty:
            df0.columns = [re.sub(pattern, "", col).strip() for col in df0.columns]
            if '股票简称' not in df0.columns:
                df0['股票简称'] = name
            df0 = normalize_columns(df0)
            df0 = coerce_numeric(df0, ['量比','换手率','振幅','市盈率(pe)','最新dde大单净额','涨停开板次数','连续涨停天数'])
            if 'a股市值(不含限售股)' in df0.columns:
                df0['a股市值(不含限售股)'] = df0['a股市值(不含限售股)'].apply(parse_market_cap)
            df0['涨停日期'] = date
            df0['持仓'] = True
            frames.append(df0.iloc[[0]])
        else:
            try:
                r2 = pywencai.get(query=f"{name}的股票代码,股票简称", query_type='stock', loop=True)
            except Exception:
                r2 = None
            code_val = None
            if isinstance(r2, pd.DataFrame) and not r2.empty:
                cols = [re.sub(pattern, "", str(c)).strip() for c in r2.columns]
                r2.columns = cols
                for _, row in r2.iterrows():
                    c = row.get('股票代码') or row.get('code')
                    n = row.get('股票简称') or row.get('股票名称')
                    if pd.notna(c) and str(n).strip() == str(name).strip():
                        code_val = str(c)
                        break
                if code_val is None:
                    v = r2.iloc[0].get('股票代码') or r2.iloc[0].get('code')
                    if pd.notna(v):
                        code_val = str(v)
            if code_val:
                try:
                    import baostock as bs
                    lg = bs.login()
                    bs_code = code_val
                    if not bs_code.startswith(('sh.','sz.','bj.')):
                        if bs_code.startswith('6'):
                            bs_code = 'sh.' + bs_code
                        elif bs_code.startswith(('0','3')):
                            bs_code = 'sz.' + bs_code
                        else:
                            bs_code = bs_code
                    rs = bs.query_history_k_data_plus(bs_code, "date,code,close,turn,pctChg", start_date=str(date), end_date=str(date), frequency="d", adjustflag="3")
                    data_list = []
                    while (rs.error_code == '0') and rs.next():
                        data_list.append(rs.get_row_data())
                    bs.logout()
                    if data_list:
                        row = data_list[0]
                        d = {
                            'code': str(row[1]).replace('sh.','').replace('sz.','').replace('bj.',''),
                            '股票简称': name,
                            '最新价': pd.to_numeric(row[2], errors='coerce'),
                            '换手率': pd.to_numeric(row[3], errors='coerce'),
                            '最新涨跌幅': pd.to_numeric(row[4], errors='coerce'),
                            '涨停日期': date,
                            '持仓': True
                        }
                        frames.append(pd.DataFrame([d]))
                except Exception:
                    pass
            if not frames:
                try:
                    r3 = pywencai.get(query=f"股票简称={name},最新价,最新涨跌幅,量比,换手率,振幅,所属概念,所属同花顺行业,市盈率,涨停开板次数,最新dde大单净额,a股市值(不含限售股)", query_type='stock', loop=True)
                except Exception:
                    r3 = None
                if isinstance(r3, pd.DataFrame) and not r3.empty:
                    r3.columns = [re.sub(pattern, "", str(c)).strip() for c in r3.columns]
                    r3 = normalize_columns(r3)
                    if '股票简称' not in r3.columns:
                        r3['股票简称'] = name
                    r3 = coerce_numeric(r3, ['量比','换手率','振幅','市盈率(pe)','最新dde大单净额','涨停开板次数','连续涨停天数'])
                    if 'a股市值(不含限售股)' in r3.columns:
                        r3['a股市值(不含限售股)'] = r3['a股市值(不含限售股)'].apply(parse_market_cap)
                    r3['涨停日期'] = date
                    r3['持仓'] = True
                    frames.append(r3.iloc[[0]])
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def out_path(filename):
    fn = str(filename)
    if fn.lower().endswith('.md'):
        return os.path.join(REPORTS_DIR, fn)
    else:
        return os.path.join(DATA_DIR, fn)

def get_market_sentiment(date):
    """获取大盘情绪数据"""
    try:
        cached = load_market_cache(date)
        def fetch_ths_all_sentiment(date):
            q = f"{date}上证指数涨跌幅:前复权,深证成指涨跌幅:前复权,创业板指涨跌幅:前复权,同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
            q_nodate = "上证指数涨跌幅:前复权,深证成指涨跌幅:前复权,创业板指涨跌幅:前复权,同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
            try:
                r = pywencai.get(query=q, query_type='zhishu', loop=True)
            except Exception:
                r = None
            if (r is None) or (isinstance(r, pd.DataFrame) and r.empty) or (isinstance(r, list) and (len(r)==0)):
                try:
                    r = pywencai.get(query=q_nodate, query_type='zhishu', loop=True)
                except Exception:
                    r = None
            def parse_df(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return None
                df = df.copy()
                df.columns = [re.sub(pattern, "", str(c)).strip() for c in df.columns]
                cols = list(df.columns)
                def find_col(tokens):
                    for c in cols:
                        n = str(c)
                        if all(t in n for t in tokens):
                            return c
                    return None
                idx_col = find_col(['指数', '简称']) or find_col(['指数', '名称']) or find_col(['指数', '代码'])
                code_col = find_col(['指数', '代码'])
                def pick_by_alias(alias_exact, tokens, is_pct=False):
                    col = find_col(tokens)
                    if col is None:
                        return np.nan
                    sub = df
                    if idx_col is not None:
                        # 优先精确匹配别名
                        sub = df[df[idx_col].astype(str) == alias_exact]
                        if sub.empty:
                            # 次选包含匹配
                            sub = df[df[idx_col].astype(str).str.contains(alias_exact.split('(')[0], na=False)]
                        if sub.empty:
                            sub = df
                    v = sub[col].iloc[0]
                    return pd.to_numeric(str(v).replace('%',''), errors='coerce') if is_pct else pd.to_numeric(str(v).replace(',',''), errors='coerce')
                def pick_by_code(code_exact, tokens, is_pct=False):
                    col = find_col(tokens)
                    if col is None:
                        return np.nan
                    sub = df
                    if code_col is not None:
                        sub = df[df[code_col].astype(str) == code_exact]
                        if sub.empty:
                            sub = df
                    v = sub[col].iloc[0]
                    return pd.to_numeric(str(v).replace('%',''), errors='coerce') if is_pct else pd.to_numeric(str(v).replace(',',''), errors='coerce')
                try:
                    d_dot = datetime.strptime(date, '%Y%m%d').strftime('%Y.%m.%d')
                except Exception:
                    d_dot = ''
                pct_tokens = ['涨跌幅', '前复权'] if d_dot == '' else ['涨跌幅', '前复权', d_dot]
                up_tokens = ['上涨家数', '家'] if d_dot == '' else ['上涨家数', '家', d_dot]
                down_tokens = ['下跌家数', '家'] if d_dot == '' else ['下跌家数', '家', d_dot]
                flat_tokens = ['平盘家数', '家'] if d_dot == '' else ['平盘家数', '家', d_dot]
                zt_tokens = ['涨停家数', '家'] if d_dot == '' else ['涨停家数', '家', d_dot]
                dt_tokens = ['跌停家数', '家'] if d_dot == '' else ['跌停家数', '家', d_dot]
                row = {
                    '上证指数涨跌幅': pick_by_code('000001.SH', ['指数'] + pct_tokens, True),
                    '深证成指涨跌幅': pick_by_code('399001.SZ', ['指数'] + pct_tokens, True),
                    '创业板指涨跌幅': pick_by_code('399006.SZ', ['指数'] + pct_tokens, True),
                    '上涨家数': pick_by_code('883957.TI', ['指数'] + up_tokens),
                    '下跌家数': pick_by_code('883957.TI', ['指数'] + down_tokens),
                    '平盘家数': pick_by_code('883957.TI', ['指数'] + flat_tokens),
                    '涨停家数': pick_by_code('883957.TI', ['指数'] + zt_tokens),
                    '跌停家数': pick_by_code('883957.TI', ['指数'] + dt_tokens),
                }
                return pd.DataFrame([row])
            if isinstance(r, pd.DataFrame):
                return parse_df(r)
            if isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                return parse_df(r[0])
            if isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                return parse_df(r['tableV1'])
            return None
        def update_market_from_query(date):
            try:
                q = f"{date}上证指数涨跌幅:前复权,深证成指涨跌幅:前复权,创业板指涨跌幅:前复权,同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
                r = pywencai.get(query=q, query_type='zhishu', loop=True)
                df0 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df0 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df0 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df0 = r['tableV1']
                if not isinstance(df0, pd.DataFrame) or df0.empty:
                    try:
                        q2 = "上证指数涨跌幅:前复权,深证成指涨跌幅:前复权,创业板指涨跌幅:前复权,同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
                        r2 = pywencai.get(query=q2, query_type='zhishu', loop=True)
                        if isinstance(r2, pd.DataFrame) and not r2.empty:
                            df0 = r2
                        elif isinstance(r2, list) and len(r2) > 0 and isinstance(r2[0], pd.DataFrame):
                            df0 = r2[0]
                        elif isinstance(r2, dict) and 'tableV1' in r2 and isinstance(r2['tableV1'], pd.DataFrame):
                            df0 = r2['tableV1']
                        if not isinstance(df0, pd.DataFrame) or df0.empty:
                            return False
                    except Exception:
                        return False
                df0 = df0.copy()
                cols_raw = list(df0.columns)
                df0.columns = [re.sub(pattern, "", str(c)).strip() for c in df0.columns]
                cols = list(df0.columns)
                code_col = next((c for c in cols if ('指数代码' in str(c))), None)
                try:
                    d_dot = datetime.strptime(date, '%Y%m%d').strftime('%Y.%m.%d')
                    d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                except Exception:
                    d_dot = ''
                    d_dash = ''
                def pick_val(row, tokens, pct=False):
                    # 优先按原始列名包含日期的精确匹配
                    for c in cols_raw:
                        n = str(c)
                        if all(t in n for t in tokens + [date]) or all(t in n for t in tokens + [d_dot]) or all(t in n for t in tokens + [d_dash]):
                            v = row[c]
                            return pd.to_numeric(str(v).replace('%',''), errors='coerce') if pct else pd.to_numeric(str(v).replace(',',''), errors='coerce')
                    # 兜底按清洗后的列名不带日期匹配
                    for c in cols:
                        n = str(c)
                        if all(t in n for t in tokens):
                            v = row[c]
                            return pd.to_numeric(str(v).replace('%',''), errors='coerce') if pct else pd.to_numeric(str(v).replace(',',''), errors='coerce')
                    return np.nan
                def row_by_code(code):
                    if code_col is None:
                        return None
                    sub = df0[df0[code_col].astype(str) == code]
                    return sub.iloc[0] if not sub.empty else None
                r_sh = row_by_code('000001.SH')
                r_sz = row_by_code('399001.SZ')
                r_cyb = row_by_code('399006.SZ')
                r_all = row_by_code('883957.TI')
                if r_all is None:
                    return False
                # 解析指数家数与指数涨跌幅
                out = pd.DataFrame([{
                    '上证指数涨跌幅': pick_val(r_sh, ['涨跌幅','前复权'], True) if r_sh is not None else np.nan,
                    '深证成指涨跌幅': pick_val(r_sz, ['涨跌幅','前复权'], True) if r_sz is not None else np.nan,
                    '创业板指涨跌幅': pick_val(r_cyb, ['涨跌幅','前复权'], True) if r_cyb is not None else np.nan,
                    '上涨家数': pick_val(r_all, ['上涨家数','家']),
                    '下跌家数': pick_val(r_all, ['下跌家数','家']),
                    '平盘家数': pick_val(r_all, ['平盘家数','家']),
                    '涨停家数': pick_val(r_all, ['涨停家数','家']),
                    '跌停家数': pick_val(r_all, ['跌停家数','家']),
                    'A股总成交额(元)': np.nan,
                }])
                # 直接从同花顺全A行取成交额（优先带日期标签的列），兼容单位（亿）
                try:
                    tokens_amt = ['成交额']
                    d_dot = ''
                    try:
                        d_dot = datetime.strptime(date, '%Y%m%d').strftime('%Y.%m.%d')
                    except Exception:
                        d_dot = ''
                    candidates = []
                    for c in cols:
                        n = str(c)
                        if all(t in n for t in tokens_amt):
                            if (d_dot == '') or (d_dot in n):
                                candidates.append(c)
                    def parse_amt(v, col_name):
                        try:
                            s = str(v)
                            unit = 1.0
                            if '亿' in s:
                                unit = 1e8
                            elif '万' in s:
                                unit = 1e4
                            s_clean = re.sub(r"[^0-9\.\-]", "", s)
                            x = pd.to_numeric(s_clean, errors='coerce')
                            if pd.notna(x):
                                return float(x) * unit
                            x2 = pd.to_numeric(str(v).replace(',',''), errors='coerce')
                            if pd.notna(x2):
                                if ('(亿)' in str(col_name)) or ('亿元' in str(col_name)):
                                    return float(x2) * 1e8
                                return float(x2)
                        except Exception:
                            pass
                        return np.nan
                    if r_all is not None and len(candidates) > 0:
                        val = parse_amt(r_all[candidates[0]], candidates[0])
                        if pd.notna(val):
                            out.loc[0, 'A股总成交额(元)'] = val
                except Exception:
                    pass
                # 若未命中，使用专用查询兜底
                if pd.isna(out.loc[0, 'A股总成交额(元)']):
                    try:
                        v = fetch_total_amt(date)
                        if pd.notna(v):
                            out.loc[0, 'A股总成交额(元)'] = float(v)
                    except Exception:
                        pass
                save_market_cache(date, out)
                return True
            except Exception:
                return False
        if update_market_from_query(date):
            cached2 = load_market_cache(date)
            if isinstance(cached2, pd.DataFrame) and not cached2.empty and is_market_complete(cached2):
                return cached2
        ths_df = fetch_ths_all_sentiment(date)
        if isinstance(ths_df, pd.DataFrame) and not ths_df.empty:
            save_market_cache(date, ths_df)
            if is_market_complete(ths_df):
                return ths_df
        def fetch_ths_counts(date):
            try:
                q = f"{date}同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
                r = pywencai.get(query=q, query_type='zhishu', loop=True)
                df0 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df0 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df0 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df0 = r['tableV1']
                if not isinstance(df0, pd.DataFrame) or df0.empty:
                    return None
                df0 = df0.copy()
                df0.columns = [re.sub(pattern, "", str(c)).strip() for c in df0.columns]
                cols = list(df0.columns)
                def pick(tokens):
                    for c in cols:
                        n = str(c)
                        if all(t in n for t in tokens):
                            return pd.to_numeric(df0[c].iloc[0], errors='coerce')
                    return np.nan
                return {
                    '上涨家数': pick(['上涨家数','家']),
                    '下跌家数': pick(['下跌家数','家']),
                    '平盘家数': pick(['平盘家数','家']),
                    '涨停家数': pick(['涨停家数','家']),
                    '跌停家数': pick(['跌停家数','家']),
                }
            except Exception:
                return None
        cnts = fetch_ths_counts(date)
        # 指数涨跌幅（AKShare优先，精确到指定日期）
        def index_pct(date, symbol):
            try:
                d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                idx = ak.stock_zh_index_daily_em(symbol=symbol)
                if isinstance(idx, pd.DataFrame) and not idx.empty:
                    cols = [str(c) for c in idx.columns]
                    date_col = next((c for c in cols if ('日期' in c) or ('date' in c.lower()) or ('时间' in c)), None)
                    pct_col = next((c for c in cols if ('涨跌幅' in c) or ('pct' in c.lower() and 'chg' in c.lower())), None)
                    close_col = next((c for c in cols if ('收盘' in c) or ('close' in c.lower())), None)
                    if date_col and pct_col:
                        rowd = idx[idx[date_col].astype(str) == d_dash]
                        if not rowd.empty:
                            return pd.to_numeric(rowd[pct_col].astype(str).str.replace('%',''), errors='coerce').iloc[0]
                    if date_col and close_col:
                        idx = idx.sort_values(date_col)
                        idx[close_col] = pd.to_numeric(idx[close_col], errors='coerce')
                        rowd = idx[idx[date_col].astype(str) == d_dash]
                        if not rowd.empty:
                            i = rowd.index[0]
                            # 找到前一行索引
                            prev_i = None
                            try:
                                prev_i = idx.index[idx.index.get_loc(i)-1]
                            except Exception:
                                prev_i = None
                            if prev_i is not None:
                                prev_close = idx.loc[prev_i, close_col]
                                now_close = idx.loc[i, close_col]
                                if pd.notna(prev_close) and pd.notna(now_close) and prev_close != 0:
                                    return (now_close/prev_close-1)*100
                return np.nan
            except Exception:
                pass
            # 回退：问财精确取指数涨跌幅
            try:
                name_map = {
                    'SH000001': '上证指数',
                    'SZ399001': '深证成指',
                    'SZ399006': '创业板指',
                }
                idx_name = name_map.get(symbol, symbol)
                code_map = {
                    'SH000001': ['SH000001','000001.SH'],
                    'SZ399001': ['SZ399001','399001.SZ'],
                    'SZ399006': ['SZ399006','399006.SZ'],
                }
                r = pywencai.get(query=f'{date}{idx_name}涨跌幅', query_type='zhishu', loop=True)
                df0 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df0 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df0 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df0 = r['tableV1']
                if isinstance(df0, pd.DataFrame) and not df0.empty:
                    df0 = df0.copy()
                    df0.columns = [re.sub(pattern, "", str(c)).strip() for c in df0.columns]
                    name_col = next((c for c in df0.columns if ('指数简称' in c) or ('指数名称' in c) or ('指数' in c)), None)
                    code_col = next((c for c in df0.columns if ('指数代码' in c)), None)
                    pct_cols = [c for c in df0.columns if ('涨跌幅' in c)]
                    if pct_cols:
                        row = pd.DataFrame()
                        if code_col and symbol in code_map:
                            codes = code_map[symbol]
                            row = df0[df0[code_col].astype(str).isin(codes)]
                        if row.empty and name_col:
                            row = df0[df0[name_col].astype(str).str.contains(idx_name, na=False)]
                        if row.empty:
                            if code_col:
                                row = df0[df0[code_col].astype(str).str.contains(symbol, na=False)]
                            if row.empty:
                                row = df0
                        v = row[pct_cols[0]].iloc[0]
                        return pd.to_numeric(str(v).replace('%',''), errors='coerce')
            except Exception:
                return np.nan
            return np.nan
        if isinstance(cnts, dict):
            row = {
                '上证指数涨跌幅': index_pct(date, 'SH000001'),
                '深证成指涨跌幅': index_pct(date, 'SZ399001'),
                '创业板指涨跌幅': index_pct(date, 'SZ399006'),
                '上涨家数': cnts.get('上涨家数'),
                '下跌家数': cnts.get('下跌家数'),
                '平盘家数': cnts.get('平盘家数'),
                '涨停家数': cnts.get('涨停家数'),
                '跌停家数': cnts.get('跌停家数'),
            }
            out = pd.DataFrame([row])
            save_market_cache(date, out)
            return out
        def get_breadth_counts(date):
            dt_cnt = np.nan
            up_cnt = np.nan
            down_cnt = np.nan
            try:
                dt_df = pywencai.get(query=f'{date}跌停股票,非st股票', loop=True)
                if isinstance(dt_df, pd.DataFrame):
                    dt_cnt = len(dt_df)
            except Exception:
                pass
            try:
                bd = pywencai.get(query=f'{date}上涨家数,下跌家数', query_type='zhishu', loop=True)
                if isinstance(bd, pd.DataFrame) and not bd.empty:
                    bd.columns = [re.sub(pattern, "", col).strip() for col in bd.columns]
                    def pick(df, tokens):
                        for col in df.columns:
                            if all(t in str(col) for t in tokens):
                                return col
                        return None
                    up_col = pick(bd, ['上涨', '家数'])
                    down_col = pick(bd, ['下跌', '家数'])
                    if up_col:
                        up_cnt = pd.to_numeric(bd[up_col].iloc[0], errors='coerce')
                    if down_col:
                        down_cnt = pd.to_numeric(bd[down_col].iloc[0], errors='coerce')
            except Exception:
                pass
            # 若仍为空，尝试直接取全市场涨跌幅并统计
            if pd.isna(up_cnt) or pd.isna(down_cnt):
                try:
                    q = pywencai.get(query=f'{date}A股股票,最新涨跌幅,非st股票', loop=True)
                    df_all = None
                    if isinstance(q, pd.DataFrame):
                        df_all = q
                    elif isinstance(q, list):
                        df_all = next((x for x in q if isinstance(x, pd.DataFrame)), None)
                    if isinstance(df_all, pd.DataFrame) and not df_all.empty:
                        df_all.columns = [re.sub(pattern, "", col).strip() for col in df_all.columns]
                        col = next((c for c in df_all.columns if '涨跌幅' in str(c)), None)
                        if col:
                            ser = pd.to_numeric(df_all[col].astype(str).str.replace('%',''), errors='coerce')
                            up_cnt = int((ser > 0).sum())
                            down_cnt = int((ser < 0).sum())
                except Exception:
                    pass
            if pd.isna(up_cnt) or pd.isna(down_cnt) or pd.isna(dt_cnt):
                try:
                    import baostock as bs
                    lg = bs.login()
                    d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                    rs = bs.query_all_stock()
                    codes = []
                    while (rs.error_code == '0') and rs.next():
                        row = rs.get_row_data()
                        codes.append(row[0])
                    u = 0
                    d = 0
                    dc = 0
                    for code in codes:
                        krs = bs.query_history_k_data_plus(code, 'date,code,pctChg', start_date=d_dash, end_date=d_dash, frequency='d', adjustflag='3')
                        if krs.error_code == '0':
                            if krs.next():
                                rr = krs.get_row_data()
                                pct = pd.to_numeric(rr[2], errors='coerce')
                                if pd.notna(pct):
                                    if pct > 0:
                                        u += 1
                                    elif pct < 0:
                                        d += 1
                                    if pct <= -9.9:
                                        dc += 1
                    bs.logout()
                    if pd.isna(up_cnt):
                        up_cnt = u
                    if pd.isna(down_cnt):
                        down_cnt = d
                    if pd.isna(dt_cnt):
                        dt_cnt = dc
                except Exception:
                    pass
            return dt_cnt, up_cnt, down_cnt
        # 直接使用“同花顺全A(沪深京)”的家数 + 指数当日涨跌幅（更稳健）
        try:
            cnts = fetch_ths_all_counts(date)
        except Exception:
            cnts = None
        row = {
            '上证指数涨跌幅': index_pct(date, 'SH000001'),
            '深证成指涨跌幅': index_pct(date, 'SZ399001'),
            '创业板指涨跌幅': index_pct(date, 'SZ399006'),
            '涨停家数': cnts.get('涨停家数') if isinstance(cnts, dict) else np.nan,
            '跌停家数': cnts.get('跌停家数') if isinstance(cnts, dict) else np.nan,
            '上涨家数': cnts.get('上涨家数') if isinstance(cnts, dict) else np.nan,
            '下跌家数': cnts.get('下跌家数') if isinstance(cnts, dict) else np.nan,
            '平盘家数': cnts.get('平盘家数') if isinstance(cnts, dict) else np.nan,
        }
        # 兜底广度统计
        if not isinstance(cnts, dict) or any(pd.isna(row.get(k)) for k in ['上涨家数','下跌家数','涨停家数','跌停家数']):
            dt_cnt, up_cnt, down_cnt = get_breadth_counts(date)
            if pd.isna(row.get('跌停家数')):
                row['跌停家数'] = dt_cnt
            if pd.isna(row.get('上涨家数')):
                row['上涨家数'] = up_cnt
            if pd.isna(row.get('下跌家数')):
                row['下跌家数'] = down_cnt
        out = pd.DataFrame([row])
        save_market_cache(date, out)
        return out
        if isinstance(res, list) and len(res) > 0:
            if isinstance(res[0], pd.DataFrame):
                idx = extract_index_pct_from_df(res[0])
                row = {
                    '上证指数涨跌幅': idx.get('上证指数涨跌幅'),
                    '深证成指涨跌幅': idx.get('深证成指涨跌幅'),
                    '创业板指涨跌幅': idx.get('创业板指涨跌幅'),
                    '涨停家数': np.nan,
                    '跌停家数': np.nan,
                    '上涨家数': np.nan,
                    '下跌家数': np.nan,
                }
                # 补充广度统计
                dt_cnt, up_cnt, down_cnt = get_breadth_counts(date)
                if pd.isna(row.get('跌停家数')):
                    row['跌停家数'] = dt_cnt
                if pd.isna(row.get('上涨家数')):
                    row['上涨家数'] = up_cnt
                if pd.isna(row.get('下跌家数')):
                    row['下跌家数'] = down_cnt
                out = pd.DataFrame([row])
                save_market_cache(date, out)
                return out
        if isinstance(res, dict):
            if 'tableV1' in res and isinstance(res['tableV1'], pd.DataFrame):
                tdf = res['tableV1'].copy()
                tdf.columns = [re.sub(pattern, "", str(c)).strip() for c in tdf.columns]
                def pick_col(cols, tokens):
                    for c in cols:
                        n = str(c)
                        if all(tok in n for tok in tokens):
                            return c
                    return None
                cols = list(tdf.columns)
                up_col = pick_col(cols, ['指数', '上涨家数'])
                down_col = pick_col(cols, ['指数', '下跌家数'])
                pct_col = pick_col(cols, ['指数', '涨跌幅'])
                # 行可能为各指数，取前三行对应上证/创业板/深证
                def val_by_index(alias):
                    idx_col = pick_col(cols, ['指数', '简称'])
                    df2 = tdf
                    if idx_col is not None:
                        df2 = tdf[tdf[idx_col].astype(str).str.contains(alias, na=False)]
                        if df2.empty:
                            df2 = tdf
                    return df2
                def take_pct(alias):
                    df2 = val_by_index(alias)
                    if pct_col is None or df2.empty:
                        return np.nan
                    return pd.to_numeric(str(df2[pct_col].iloc[0]).replace('%',''), errors='coerce')
                def take_int(col):
                    if col is None:
                        return np.nan
                    return pd.to_numeric(tdf[col].iloc[0], errors='coerce')
                # 涨跌停家数可能在后续行，以简表形式出现
                zt_col = pick_col(cols, ['指数', '涨停家数'])
                dt_col = pick_col(cols, ['指数', '跌停家数'])
                row = {
                    '上证指数涨跌幅': take_pct('上证'),
                    '深证成指涨跌幅': take_pct('深证'),
                    '创业板指涨跌幅': take_pct('创业'),
                    '上涨家数': take_int(up_col),
                    '下跌家数': take_int(down_col),
                    '涨停家数': take_int(zt_col),
                    '跌停家数': take_int(dt_col),
                }
            else:
                keys = ['涨停家数','跌停家数','上涨家数','下跌家数','上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅']
                row = {k: res.get(k) for k in keys}
                for k in ['涨停家数','跌停家数','上涨家数','下跌家数']:
                    if row.get(k) is not None:
                        try:
                            row[k] = float(str(row[k]).replace(',',''))
                        except Exception:
                            row[k] = np.nan
                for k in ['上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅']:
                    if row.get(k) is not None:
                        try:
                            row[k] = float(str(row[k]).replace('%',''))
                        except Exception:
                            row[k] = np.nan
            dt_cnt, up_cnt, down_cnt = get_breadth_counts(date)
            if pd.isna(row.get('跌停家数')):
                row['跌停家数'] = dt_cnt
            if pd.isna(row.get('上涨家数')):
                row['上涨家数'] = up_cnt
            if pd.isna(row.get('下跌家数')):
                row['下跌家数'] = down_cnt
            dt_cnt, up_cnt, down_cnt = get_breadth_counts(date)
            if pd.isna(row.get('跌停家数')):
                row['跌停家数'] = dt_cnt
            if pd.isna(row.get('上涨家数')):
                row['上涨家数'] = up_cnt
            if pd.isna(row.get('下跌家数')):
                row['下跌家数'] = down_cnt
            out = pd.DataFrame([row])
            save_market_cache(date, out)
            return out
        # 回退：使用 akshare 获取指数当日涨跌幅
        try:
            sz = ak.stock_zh_index_daily_em(symbol='SZ399001')
            sh = ak.stock_zh_index_daily_em(symbol='SH000001')
            cyb = ak.stock_zh_index_daily_em(symbol='SZ399006')
            d_str = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            def get_pct(df):
                row = df[df['日期'] == d_str]
                if row.empty:
                    return np.nan
                return pd.to_numeric(row['涨跌幅'].astype(str).str.replace('%',''), errors='coerce').iloc[0]
            row = {
                '上证指数涨跌幅': get_pct(sh),
                '深证成指涨跌幅': get_pct(sz),
                '创业板指涨跌幅': get_pct(cyb),
                '涨停家数': np.nan,
                '跌停家数': np.nan,
                '上涨家数': np.nan,
                '下跌家数': np.nan,
            }
            return pd.DataFrame([row])
        except Exception:
            pass
        row = {
            '上证指数涨跌幅': index_pct(date, 'SH000001'),
            '深证成指涨跌幅': index_pct(date, 'SZ399001'),
            '创业板指涨跌幅': index_pct(date, 'SZ399006'),
            '涨停家数': np.nan,
            '跌停家数': np.nan,
            '上涨家数': np.nan,
            '下跌家数': np.nan,
        }
        dt_cnt, up_cnt, down_cnt = get_breadth_counts(date)
        row['跌停家数'] = dt_cnt
        row['上涨家数'] = up_cnt
        row['下跌家数'] = down_cnt
        out = pd.DataFrame([row])
        save_market_cache(date, out)
        return out
        # 备用方案：使用 AKShare 获取指数与市场家数（尽力而为）
        try:
            d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            spot = ak.stock_zh_a_spot_em()
            up = down = np.nan
            if isinstance(spot, pd.DataFrame) and '涨跌幅' in spot.columns:
                s = pd.to_numeric(spot['涨跌幅'].astype(str).str.replace('%',''), errors='coerce')
                up = int((s > 0).sum())
                down = int((s < 0).sum())
            def idx_pct(sym):
                try:
                    idx = ak.stock_zh_index_daily_em(symbol=sym)
                    if isinstance(idx, pd.DataFrame) and not idx.empty:
                        # 兼容列名：日期、涨跌幅 或 close 等
                        if '日期' in idx.columns and '涨跌幅' in idx.columns:
                            rowd = idx[idx['日期'] == d_dash]
                            if not rowd.empty:
                                return pd.to_numeric(rowd['涨跌幅'].astype(str).str.replace('%',''), errors='coerce').iloc[0]
                        # 若无“涨跌幅”列，则以收盘价计算近一日变动（简化）
                        if '日期' in idx.columns and '收盘' in idx.columns:
                            idx = idx.sort_values('日期')
                            idx['收盘'] = pd.to_numeric(idx['收盘'], errors='coerce')
                            rowd = idx[idx['日期'] == d_dash]
                            if not rowd.empty:
                                i = rowd.index[0]
                                if i > idx.index.min():
                                    prev_close = idx.loc[idx.index[idx.index.get_loc(i)-1], '收盘']
                                    now_close = idx.loc[i, '收盘']
                                    if pd.notna(prev_close) and pd.notna(now_close) and prev_close != 0:
                                        return (now_close/prev_close-1)*100
                except Exception:
                    return np.nan
                return np.nan
            zt_cnt = np.nan
            dt_cnt2 = np.nan
            try:
                # 尝试使用东方财富涨跌停池（若不可用则保持 NaN）
                zt_pool = ak.stock_zt_pool_em(date=d_dash)
                if isinstance(zt_pool, pd.DataFrame):
                    zt_cnt = len(zt_pool)
            except Exception:
                pass
            try:
                dt_pool = ak.stock_dt_pool_em(date=d_dash)
                if isinstance(dt_pool, pd.DataFrame):
                    dt_cnt2 = len(dt_pool)
            except Exception:
                pass
            row2 = {
                '上证指数涨跌幅': idx_pct('SH000001'),
                '深证成指涨跌幅': idx_pct('SZ399001'),
                '创业板指涨跌幅': idx_pct('SZ399006'),
                '涨停家数': zt_cnt,
                '跌停家数': dt_cnt2,
                '上涨家数': up,
                '下跌家数': down,
            }
            out2 = pd.DataFrame([row2])
            save_market_cache(date, out2)
            return out2
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"获取大盘情绪数据失败: {e}")
        return None

def recent_dates(end_date, days=3):
    try:
        dt = datetime.strptime(end_date, '%Y%m%d')
        return [(dt - pd.Timedelta(days=i)).strftime('%Y%m%d') for i in range(days)]
    except Exception:
        return [end_date]

def trading_recent_dates(end_date, days=3):
    try:
        p_all = all_market_cache_path()
        if os.path.exists(p_all):
            hist = pd.read_csv(p_all, encoding='utf-8-sig')
            if '日期' in hist.columns:
                ds = sorted([str(x) for x in hist['日期'].astype(str).tolist()])
                ds = [d for d in ds if d <= end_date]
                if ds:
                    return ds[-min(days, len(ds)):]
    except Exception:
        pass
    return recent_dates(end_date, days)

def fetch_board_money_pct(date):
    try:
        metrics_primary = ['当日成交额','较上一交易日成交额','近三交易日成交额','涨跌幅','主力净流入']
        metrics_fallback = ['总成交额','涨跌幅','主力净流入']
        def try_query(qdate):
            qs = [
                f"{qdate}" + ','.join(metrics_primary),
                f"{qdate}指数" + ',指数'.join(metrics_primary),
                f"{qdate}" + ','.join([x + '(元)' if '成交额' in x else x for x in metrics_primary])
            ]
            for q in qs:
                r = pywencai.get(query=q, query_type='zhishu', loop=True)
                df0 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df0 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df0 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df0 = r['tableV1']
                if isinstance(df0, pd.DataFrame) and not df0.empty:
                    return df0
    except Exception:
        return None
    return None

def update_market_from_probe(date):
    try:
        p = os.path.join(os.path.dirname(__file__), f'probe_zhishu_{date}.csv')
        if not os.path.exists(p):
            return False
        df = pd.read_csv(p, encoding='utf-8-sig')
        df.columns = [re.sub(pattern, '', str(c)).strip() for c in df.columns]
        code_col = next((c for c in df.columns if ('指数代码' in str(c))), None)
        pct_col = next((c for c in df.columns if ('涨跌幅' in str(c)) and ('前复权' in str(c))), None)
        up_col = next((c for c in df.columns if ('上涨家数' in str(c)) and ('家' in str(c))), None)
        down_col = next((c for c in df.columns if ('下跌家数' in str(c)) and ('家' in str(c))), None)
        flat_col = next((c for c in df.columns if ('平盘家数' in str(c)) and ('家' in str(c))), None)
        zt_col = next((c for c in df.columns if ('涨停家数' in str(c)) and ('家' in str(c))), None)
        dt_col = next((c for c in df.columns if ('跌停家数' in str(c)) and ('家' in str(c))), None)
        def val_code(c):
            sub = df[df[code_col].astype(str) == c] if code_col is not None else pd.DataFrame()
            return sub
        def num(x):
            return pd.to_numeric(str(x).replace('%','').replace(',',''), errors='coerce')
        sh = val_code('000001.SH')
        sz = val_code('399001.SZ')
        cyb = val_code('399006.SZ')
        ths = val_code('883957.TI')
        row = {
            '上证指数涨跌幅': num(sh[pct_col].iloc[0]) if (pct_col and not sh.empty) else np.nan,
            '深证成指涨跌幅': num(sz[pct_col].iloc[0]) if (pct_col and not sz.empty) else np.nan,
            '创业板指涨跌幅': num(cyb[pct_col].iloc[0]) if (pct_col and not cyb.empty) else np.nan,
            '上涨家数': num(ths[up_col].iloc[0]) if (up_col and not ths.empty) else np.nan,
            '下跌家数': num(ths[down_col].iloc[0]) if (down_col and not ths.empty) else np.nan,
            '平盘家数': num(ths[flat_col].iloc[0]) if (flat_col and not ths.empty) else np.nan,
            '涨停家数': num(ths[zt_col].iloc[0]) if (zt_col and not ths.empty) else np.nan,
            '跌停家数': num(ths[dt_col].iloc[0]) if (dt_col and not ths.empty) else np.nan,
        }
        save_market_cache(date, pd.DataFrame([row]))
        return True
    except Exception:
        return False

def ingest_market_from_fetch(date):
    try:
        p = os.path.join(os.path.dirname(__file__), 'cache_market', f'市场情绪_{date}.csv')
        if not os.path.exists(p):
            return False
        df = pd.read_csv(p, encoding='utf-8-sig')
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cols = df.columns.tolist()
        def pick(tokens):
            for c in cols:
                cc = str(c)
                ok = True
                for t in tokens:
                    if t not in cc:
                        ok = False
                        break
                if ok:
                    return c
            return None
        up_col = pick(['上涨家数'])
        down_col = pick(['下跌家数'])
        flat_col = pick(['平盘家数'])
        zt_col = pick(['涨停家数'])
        dt_col = pick(['跌停家数'])
        sz_col = pick(['上证', '涨跌幅'])
        shen_col = pick(['深证', '涨跌幅'])
        cyb_col = pick(['创业板', '涨跌幅'])
        out = pd.DataFrame([{ 
            '上证指数涨跌幅': pd.to_numeric(str(df[sz_col].iloc[0]).replace('%',''), errors='coerce') if sz_col else np.nan,
            '深证成指涨跌幅': pd.to_numeric(str(df[shen_col].iloc[0]).replace('%',''), errors='coerce') if shen_col else np.nan,
            '创业板指涨跌幅': pd.to_numeric(str(df[cyb_col].iloc[0]).replace('%',''), errors='coerce') if cyb_col else np.nan,
            '上涨家数': pd.to_numeric(df[up_col].iloc[0], errors='coerce') if up_col else np.nan,
            '下跌家数': pd.to_numeric(df[down_col].iloc[0], errors='coerce') if down_col else np.nan,
            '平盘家数': pd.to_numeric(df[flat_col].iloc[0], errors='coerce') if flat_col else np.nan,
            '涨停家数': pd.to_numeric(df[zt_col].iloc[0], errors='coerce') if zt_col else np.nan,
            '跌停家数': pd.to_numeric(df[dt_col].iloc[0], errors='coerce') if dt_col else np.nan,
        }])
        for k, sym in [('上证指数涨跌幅','SH000001'),('深证成指涨跌幅','SZ399001'),('创业板指涨跌幅','SZ399006')]:
            if pd.isna(out.loc[out.index[0], k]):
                out.loc[out.index[0], k] = index_pct(date, sym)
        if pd.isna(out.loc[out.index[0], '上涨家数']) or pd.isna(out.loc[out.index[0], '下跌家数']) or pd.isna(out.loc[out.index[0], '涨停家数']) or pd.isna(out.loc[out.index[0], '跌停家数']):
            dt_cnt, up_cnt, down_cnt = get_breadth_counts(date)
            if pd.isna(out.loc[out.index[0], '跌停家数']):
                out.loc[out.index[0], '跌停家数'] = dt_cnt
            if pd.isna(out.loc[out.index[0], '上涨家数']):
                out.loc[out.index[0], '上涨家数'] = up_cnt
            if pd.isna(out.loc[out.index[0], '下跌家数']):
                out.loc[out.index[0], '下跌家数'] = down_cnt
        save_market_cache(date, out)
        return True
    except Exception:
        return False

def fetch_ths_all_counts(date):
    try:
        q = f"{date}同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
        r = pywencai.get(query=q, query_type='zhishu', loop=True)
        df0 = None
        if isinstance(r, pd.DataFrame) and not r.empty:
            df0 = r
        elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
            df0 = r[0]
        elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
            df0 = r['tableV1']
        if (not isinstance(df0, pd.DataFrame)) or df0.empty:
            q2 = "同花顺全A(沪深京)的上涨家数,下跌家数,平盘家数,涨停家数,跌停家数"
            r2 = pywencai.get(query=q2, query_type='zhishu', loop=True)
            if isinstance(r2, pd.DataFrame) and not r2.empty:
                df0 = r2
            elif isinstance(r2, list) and len(r2) > 0 and isinstance(r2[0], pd.DataFrame):
                df0 = r2[0]
            elif isinstance(r2, dict) and 'tableV1' in r2 and isinstance(r2['tableV1'], pd.DataFrame):
                df0 = r2['tableV1']
        if not isinstance(df0, pd.DataFrame) or df0.empty:
            return None
        df0 = df0.copy()
        df0.columns = [re.sub(pattern, '', str(c)).strip() for c in df0.columns]
        code_col = next((c for c in df0.columns if ('指数代码' in str(c))), None)
        up_col = next((c for c in df0.columns if ('上涨家数' in str(c)) and ('家' in str(c))), None)
        down_col = next((c for c in df0.columns if ('下跌家数' in str(c)) and ('家' in str(c))), None)
        flat_col = next((c for c in df0.columns if ('平盘家数' in str(c)) and ('家' in str(c))), None)
        zt_col = next((c for c in df0.columns if ('涨停家数' in str(c)) and ('家' in str(c))), None)
        dt_col = next((c for c in df0.columns if ('跌停家数' in str(c)) and ('家' in str(c))), None)
        def num(x):
            return pd.to_numeric(str(x).replace('%','').replace(',',''), errors='coerce')
        sub = df0[df0[code_col].astype(str) == '883957.TI'] if code_col is not None else df0
        if sub.empty:
            sub = df0
        return {
            '上涨家数': num(sub[up_col].iloc[0]) if up_col else np.nan,
            '下跌家数': num(sub[down_col].iloc[0]) if down_col else np.nan,
            '平盘家数': num(sub[flat_col].iloc[0]) if flat_col else np.nan,
            '涨停家数': num(sub[zt_col].iloc[0]) if zt_col else np.nan,
            '跌停家数': num(sub[dt_col].iloc[0]) if dt_col else np.nan,
        }
    except Exception:
        return None

def update_market_counts(date):
    try:
        cnts = fetch_ths_all_counts(date)
        if not isinstance(cnts, dict):
            cnts = {}
        p_all = all_market_cache_path()
        if os.path.exists(p_all):
            try:
                base = pd.read_csv(p_all, encoding='utf-8-sig')
            except Exception:
                base = pd.DataFrame()
        else:
            base = pd.DataFrame()
        cols = ['日期','上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅','上涨家数','下跌家数','平盘家数','涨停家数','跌停家数','情绪分数']
        for c in cols:
            if c not in base.columns:
                base[c] = np.nan
        base['日期'] = base['日期'].astype(str)
        mask = base['日期'] == str(date)
        if any(mask):
            idxs = base.index[mask]
            for k in ['上涨家数','下跌家数','平盘家数','涨停家数','跌停家数']:
                if k in cnts:
                    base.loc[idxs, k] = cnts.get(k)
            base.loc[idxs, '上证指数涨跌幅'] = index_pct(date, 'SH000001')
            base.loc[idxs, '深证成指涨跌幅'] = index_pct(date, 'SZ399001')
            base.loc[idxs, '创业板指涨跌幅'] = index_pct(date, 'SZ399006')
            try:
                up = pd.to_numeric(base.loc[idxs, '上涨家数'], errors='coerce').iloc[0]
                down = pd.to_numeric(base.loc[idxs, '下跌家数'], errors='coerce').iloc[0]
                zt = pd.to_numeric(base.loc[idxs, '涨停家数'], errors='coerce').iloc[0]
                dt = pd.to_numeric(base.loc[idxs, '跌停家数'], errors='coerce').iloc[0]
                total = up + down if pd.notna(up) and pd.notna(down) else np.nan
                up_ratio = (up / (total + 1e-5)) if pd.notna(total) else np.nan
                sentiment = np.nan
                if pd.notna(up_ratio) and pd.notna(zt) and pd.notna(dt):
                    sentiment = (up_ratio * 0.4 + min(max(zt / 100, 0), 1) * 0.3 + (1 - min(max(dt / 50, 0), 1)) * 0.3) * 100
                base.loc[idxs, '情绪分数'] = sentiment
            except Exception:
                pass
        else:
            row = {
                '日期': str(date),
                '上证指数涨跌幅': index_pct(date, 'SH000001'),
                '深证成指涨跌幅': index_pct(date, 'SZ399001'),
                '创业板指涨跌幅': index_pct(date, 'SZ399006'),
                '上涨家数': cnts.get('上涨家数'),
                '下跌家数': cnts.get('下跌家数'),
                '平盘家数': cnts.get('平盘家数'),
                '涨停家数': cnts.get('涨停家数'),
                '跌停家数': cnts.get('跌停家数'),
                '情绪分数': np.nan,
            }
            base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
        base = base[cols]
        base.to_csv(p_all, index=False, encoding='utf-8-sig')
        return True
    except Exception:
        return False
        def try_query_nodate():
            r = pywencai.get(query=','.join(metrics_primary), query_type='zhishu', loop=True)
            df0 = None
            if isinstance(r, pd.DataFrame) and not r.empty:
                df0 = r
            elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                df0 = r[0]
            elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                df0 = r['tableV1']
            return df0
        df = try_query_nodate()
        if (not isinstance(df, pd.DataFrame)) or df.empty:
            df = try_query(date)
        if (not isinstance(df, pd.DataFrame)) or df.empty:
            try:
                d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                d_dash = date
            df = try_query(d_dash)
        # 如果主查询失败，回退至总成交额
        if (not isinstance(df, pd.DataFrame)) or df.empty:
            def try_fallback(qdate):
                r = pywencai.get(query=f"{qdate}" + ','.join(metrics_fallback), query_type='zhishu', loop=True)
                df1 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df1 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df1 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df1 = r['tableV1']
                return df1
            df = try_fallback(date)
            if (not isinstance(df, pd.DataFrame)) or df.empty:
                try:
                    d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                except Exception:
                    d_dash = date
                df = try_fallback(d_dash)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        df = df.copy()
        df.columns = [re.sub(pattern, "", str(c)).strip() for c in df.columns]
        code_col = next((c for c in df.columns if ('指数代码' in str(c))), None)
        name_col = next((c for c in df.columns if ('指数简称' in str(c)) or ('指数名称' in str(c))), None)
        # 主列匹配：当日成交额、较上一交易日成交额、近三交易日成交额
        amt_today_col = next((c for c in df.columns if ('当日' in str(c) and '成交额' in str(c))), None)
        if amt_today_col is None:
            amt_today_col = next((c for c in df.columns if ('成交额' in str(c))), None)
        amt_delta_col = next((c for c in df.columns if ('较上' in str(c) and '成交额' in str(c))), None)
        amt_3day_col = next((c for c in df.columns if ('近三' in str(c) and '成交额' in str(c))), None)
        # 回退：总成交额
        amt_total_col = next((c for c in df.columns if ('总成交额' in str(c) or ('成交额' in str(c) and amt_today_col is None))), None)
        pct_col = next((c for c in df.columns if ('涨跌幅' in str(c))), None)
        inflow_col = next((c for c in df.columns if (('主力' in str(c)) and (('净流' in str(c)) or ('资金净流入' in str(c)) or ('净额' in str(c))))), None)
        def parse_amt(val, colname):
            s = str(val)
            mul = 1.0
            cn = str(colname)
            if ('亿' in cn) or ('亿元' in cn) or ('亿' in s):
                mul = 1e8
            elif ('万' in cn) or ('万元' in cn) or ('万' in s):
                mul = 1e4
            ts = re.sub(r"[^0-9\.-]", "", s)
            v = pd.to_numeric(ts, errors='coerce')
            return v*mul if pd.notna(v) else np.nan
        def series_parse(col):
            if col is None:
                return pd.Series([np.nan]*len(df))
            return df[col].apply(lambda x: parse_amt(x, col))
        idx_name = df[name_col].astype(str) if name_col is not None else (df[code_col].astype(str) if code_col is not None else pd.Series(['']*len(df)))
        out = pd.DataFrame({
            '日期': [date]*len(df),
            '指数代码': df[code_col].astype(str) if code_col is not None else '',
            '指数名称': idx_name,
            '成交额(元)': series_parse(amt_today_col) if amt_today_col is not None else series_parse(amt_total_col),
            '涨跌幅(%)': pd.to_numeric(df[pct_col].astype(str).str.replace('%',''), errors='coerce') if pct_col is not None else np.nan,
            '主力净流入(元)': series_parse(inflow_col) if inflow_col is not None else pd.Series([np.nan]*len(df)),
            '较上一交易日成交额(元)': series_parse(amt_delta_col),
            '近三交易日成交额(元)': series_parse(amt_3day_col),
        })
        out = out.dropna(subset=['指数名称'])
        return out
    except Exception:
        return None

def fetch_board_money_pct_range(dates):
    try:
        if not dates:
            return None
        start = dates[0]
        end = dates[-1]
        def fmt_dot(d):
            try:
                return datetime.strptime(d, '%Y%m%d').strftime('%Y.%m.%d')
            except Exception:
                return d
        def fmt_dash(d):
            try:
                return datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                return d
        q = f"{fmt_dot(start)}到{fmt_dot(end)}总成交额,涨跌幅"
        r = pywencai.get(query=q, query_type='zhishu', loop=True)
        df = None
        if isinstance(r, pd.DataFrame) and not r.empty:
            df = r
        elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
            df = r[0]
        elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
            df = r['tableV1']
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        cols = [str(c).strip() for c in df.columns]
        code_col = next((c for c in cols if '指数代码' in c), None)
        name_col = next((c for c in cols if ('指数简称' in c) or ('指数名称' in c)), None)
        # 找到每个日期对应的成交额/涨跌幅列（列名一般含日期）
        date_map = {}
        for d in dates:
            cand = [fmt_dot(d), fmt_dash(d)]
            amt_col = next((c for c in cols if ('成交额' in c) and any(x in c for x in cand)), None)
            pct_col = next((c for c in cols if ('涨跌幅' in c) and any(x in c for x in cand)), None)
            date_map[d] = {'amt': amt_col, 'pct': pct_col}
        if not date_map:
            return None
        rows = []
        for _, row in df.iterrows():
            code = str(row[code_col]) if code_col else ''
            name = str(row[name_col]) if name_col else ''
            for d, cols_map in date_map.items():
                amt_col = cols_map.get('amt')
                pct_col = cols_map.get('pct')
                amt = pd.to_numeric(str(row[amt_col]).replace(',',''), errors='coerce') if amt_col else np.nan
                pct = pd.to_numeric(str(row[pct_col]).replace('%',''), errors='coerce') if pct_col else np.nan
                rows.append({'日期': d, '指数代码': code, '指数名称': name, '成交额(元)': amt, '涨跌幅(%)': pct})
        out = pd.DataFrame(rows)
        return out
    except Exception:
        return None

def board_csv_path():
    return os.path.join(DATA_DIR, '板块资金与涨幅.csv')

def save_board_money_pct(dates):
    frames = []
    for d in dates:
        df = fetch_board_money_pct(d)
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)
        time.sleep(0.5)
    if (len(frames) < len(dates)) and len(dates) > 1:
        df_range = fetch_board_money_pct_range(dates)
        if isinstance(df_range, pd.DataFrame) and not df_range.empty:
            frames.append(df_range)
        if frames:
            agg = pd.concat(frames, ignore_index=True)
        # compute per-index deltas and rolling for all boards
        try:
            agg['日期'] = agg['日期'].astype(str)
            # prefer code, fallback to name
            key_col = '指数代码' if '指数代码' in agg.columns else '指数名称'
            agg['成交额(元)'] = pd.to_numeric(agg['成交额(元)'], errors='coerce')
            agg = agg.sort_values([key_col, '日期'])
            computed_diff = agg.groupby(key_col)['成交额(元)'].diff()
            # preserve precomputed deltas from ingest when present
            if '较上一交易日成交额(元)' in agg.columns:
                agg['较上一交易日成交额(元)'] = agg['较上一交易日成交额(元)'].fillna(computed_diff)
            else:
                agg['较上一交易日成交额(元)'] = computed_diff
            agg['近三交易日成交额(元)'] = (
                agg.groupby(key_col)['成交额(元)']
                   .rolling(3, min_periods=1)
                   .sum()
                   .reset_index(level=0, drop=True)
            )
        except Exception:
            pass
        def fetch_single_index_metrics(d, name):
            try:
                q1 = f"{d}{name}当日成交额,较上一交易日成交额,近三交易日成交额,涨跌幅,主力净流入"
                r = pywencai.get(query=q1, query_type='zhishu', loop=True)
                df0 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df0 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df0 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df0 = r['tableV1']
                if (not isinstance(df0, pd.DataFrame)) or df0.empty:
                    try:
                        d2 = datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
                    except Exception:
                        d2 = d
                    q2 = f"{d2}{name}当日成交额,较上一交易日成交额,近三交易日成交额,涨跌幅,主力净流入"
                    r2 = pywencai.get(query=q2, query_type='zhishu', loop=True)
                    if isinstance(r2, pd.DataFrame) and not r2.empty:
                        df0 = r2
                    elif isinstance(r2, list) and len(r2) > 0 and isinstance(r2[0], pd.DataFrame):
                        df0 = r2[0]
                    elif isinstance(r2, dict) and 'tableV1' in r2 and isinstance(r2['tableV1'], pd.DataFrame):
                        df0 = r2['tableV1']
                if (not isinstance(df0, pd.DataFrame)) or df0.empty:
                    return None
                df0 = df0.copy()
                df0.columns = [re.sub(pattern, "", str(c)).strip() for c in df0.columns]
                code_col = next((c for c in df0.columns if ('指数代码' in str(c))), None)
                name_col = next((c for c in df0.columns if ('指数简称' in str(c)) or ('指数名称' in str(c))), None)
                amt_today_col = next((c for c in df0.columns if ('当日' in str(c) and '成交额' in str(c))), None)
                amt_delta_col = next((c for c in df0.columns if ('较上' in str(c) and '成交额' in str(c))), None)
                amt_3day_col = next((c for c in df0.columns if ('近三' in str(c) and '成交额' in str(c))), None)
                pct_col = next((c for c in df0.columns if ('涨跌幅' in str(c))), None)
                inflow_col = next((c for c in df0.columns if (('主力' in str(c)) and (('净流' in str(c)) or ('资金净流入' in str(c)) or ('净额' in str(c))))), None)
                def parse_amt_val(x, cn):
                    s = str(x)
                    mul = 1.0
                    if ('亿' in cn) or ('亿元' in cn) or ('亿' in s):
                        mul = 1e8
                    elif ('万' in cn) or ('万元' in cn) or ('万' in s):
                        mul = 1e4
                    ts = re.sub(r"[^0-9\.-]", "", s)
                    v = pd.to_numeric(ts, errors='coerce')
                    return v*mul if pd.notna(v) else np.nan
                amt_today = parse_amt_val(df0[amt_today_col].iloc[0] if amt_today_col is not None else (df0[amt_3day_col].iloc[0] if amt_3day_col is not None else np.nan), amt_today_col or '')
                amt_delta = parse_amt_val(df0[amt_delta_col].iloc[0] if amt_delta_col is not None else np.nan, amt_delta_col or '')
                amt_three = parse_amt_val(df0[amt_3day_col].iloc[0] if amt_3day_col is not None else np.nan, amt_3day_col or '')
                pct_val = pd.to_numeric(str(df0[pct_col].iloc[0]).replace('%',''), errors='coerce') if pct_col is not None else np.nan
                inflow_val = parse_amt_val(df0[inflow_col].iloc[0] if inflow_col is not None else np.nan, inflow_col or '')
                return pd.DataFrame({
                    '日期': [d],
                    '指数代码': [str(df0[code_col].iloc[0]) if code_col is not None else ''],
                    '指数名称': [str(df0[name_col].iloc[0]) if name_col is not None else name],
                    '成交额(元)': [amt_today],
                    '涨跌幅(%)': [pct_val],
                    '主力净流入(元)': [inflow_val],
                    '较上一交易日成交额(元)': [amt_delta],
                    '近三交易日成交额(元)': [amt_three],
                })
            except Exception:
                return None
        core_names = ['同花顺全A(沪深京)','上证指数','深证成指','创业板指','中证1000']
        def ak_symbol_for(name):
            if name == '上证指数':
                return 'SH000001'
            if name == '深证成指':
                return 'SZ399001'
            if name == '创业板指':
                return 'SZ399006'
            if name == '中证1000':
                return 'SH000852'
            return None
        def ak_index_metrics(d, name):
            sym = ak_symbol_for(name)
            if sym is None:
                return None
            try:
                df = ak.stock_zh_index_daily_em(symbol=sym)
                d_str = datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                return None
            try:
                row = df[df['日期'] == d_str]
                if row.empty:
                    return None
                amt = pd.to_numeric(row['成交额'], errors='coerce').iloc[0] if '成交额' in row.columns else np.nan
                pct = pd.to_numeric(str(row['涨跌幅'].iloc[0]).replace('%',''), errors='coerce') if '涨跌幅' in row.columns else np.nan
                return pd.DataFrame({
                    '日期': [d],
                    '指数代码': [sym.replace('SH','SH').replace('SZ','SZ')],
                    '指数名称': [name],
                    '成交额(元)': [amt],
                    '涨跌幅(%)': [pct],
                    '主力净流入(元)': [np.nan],
                })
            except Exception:
                return None
        for d in dates:
            for nm in core_names:
                mask = (agg['日期'].astype(str) == str(d)) & (agg['指数名称'].astype(str) == nm)
                need = True
                if any(mask):
                    sub = agg.loc[agg.index[mask]]
                    vals = pd.to_numeric(sub['成交额(元)'], errors='coerce')
                    if not sub.empty and vals.notna().any():
                        need = False
                if need:
                    extra = fetch_single_index_metrics(d, nm)
                    if isinstance(extra, pd.DataFrame) and not extra.empty:
                        agg = pd.concat([agg, extra], ignore_index=True)
                    else:
                        extra2 = ak_index_metrics(d, nm)
                        if isinstance(extra2, pd.DataFrame) and not extra2.empty:
                            agg = pd.concat([agg, extra2], ignore_index=True)
        def fetch_board_total(d):
            try:
                q = f"{d}同花顺全A(沪深京)总成交额,涨跌幅"
                r = pywencai.get(query=q, query_type='zhishu', loop=True)
                df0 = None
                if isinstance(r, pd.DataFrame) and not r.empty:
                    df0 = r
                elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
                    df0 = r[0]
                elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
                    df0 = r['tableV1']
                if not isinstance(df0, pd.DataFrame) or df0.empty:
                    try:
                        d2 = datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
                    except Exception:
                        d2 = d
                    r2 = pywencai.get(query=f"{d2}同花顺全A(沪深京)总成交额,涨跌幅", query_type='zhishu', loop=True)
                    if isinstance(r2, pd.DataFrame) and not r2.empty:
                        df0 = r2
                    elif isinstance(r2, list) and len(r2)>0 and isinstance(r2[0], pd.DataFrame):
                        df0 = r2[0]
                    elif isinstance(r2, dict) and 'tableV1' in r2 and isinstance(r2['tableV1'], pd.DataFrame):
                        df0 = r2['tableV1']
                if not isinstance(df0, pd.DataFrame) or df0.empty:
                    return None
                df0 = df0.copy()
                df0.columns = [re.sub(pattern, "", str(c)).strip() for c in df0.columns]
                code_col = next((c for c in df0.columns if ('指数代码' in str(c))), None)
                name_col = next((c for c in df0.columns if ('指数简称' in str(c)) or ('指数名称' in str(c))), None)
                amt_col = next((c for c in df0.columns if ('成交额' in str(c))), None)
                pct_col = next((c for c in df0.columns if ('涨跌幅' in str(c))), None)
                name = '同花顺全A(沪深京)'
                return pd.DataFrame({
                    '日期': [d],
                    '指数代码': [str(df0[code_col].iloc[0]) if code_col is not None else ''],
                    '指数名称': [name],
                    '成交额(元)': [pd.to_numeric(str(df0[amt_col].iloc[0]).replace(',',''), errors='coerce') if amt_col is not None else np.nan],
                    '涨跌幅(%)': [pd.to_numeric(str(df0[pct_col].iloc[0]).replace('%',''), errors='coerce') if pct_col is not None else np.nan],
                    '主力净流入(元)': [np.nan]
                })
            except Exception:
                return None
        try:
            if os.path.exists(board_csv_path()):
                base = pd.read_csv(board_csv_path(), encoding='utf-8-sig')
                # 兼容旧列名
                if '指数' in base.columns and '指数名称' not in base.columns:
                    base = base.rename(columns={'指数': '指数名称'})
                if '指数代码' not in base.columns:
                    base['指数代码'] = ''
                base = base[~(base['日期'].astype(str).isin([str(x) for x in dates]))]
                agg = pd.concat([base, agg], ignore_index=True)
        except Exception:
            pass
        for d in dates:
            sub = agg[agg['日期'].astype(str) == str(d)]
            s = pd.to_numeric(sub['成交额(元)'], errors='coerce').sum() if '成交额(元)' in sub.columns and not sub.empty else np.nan
            if (pd.isna(s)) or (float(s) == 0.0):
                extra = fetch_board_total(d)
                if isinstance(extra, pd.DataFrame) and not extra.empty:
                    agg = pd.concat([agg, extra], ignore_index=True)
        for c in ['主力净流入(元)','较上一交易日成交额(元)','近三交易日成交额(元)']:
            if c not in agg.columns:
                agg[c] = np.nan
        try:
            agg['日期'] = agg['日期'].astype(str)
            amt_series = pd.to_numeric(agg['成交额(元)'], errors='coerce')
            delta_calc = amt_series.groupby(agg['指数名称']).diff()
            roll_calc = amt_series.groupby(agg['指数名称']).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
            agg['较上一交易日成交额(元)'] = agg['较上一交易日成交额(元)'].where(pd.notna(agg['较上一交易日成交额(元)']), delta_calc)
            agg['近三交易日成交额(元)'] = agg['近三交易日成交额(元)'].where(pd.notna(agg['近三交易日成交额(元)']), roll_calc)
            totals = agg.groupby('日期')['成交额(元)'].apply(lambda s: pd.to_numeric(s, errors='coerce').sum()).sort_index()
            dates_sorted = list(totals.index)
            totals_rolling = totals.rolling(3, min_periods=1).sum()
            totals_delta = totals.diff()
            for d in dates_sorted:
                total_val = totals.loc[d]
                roll_val = totals_rolling.loc[d]
                delta_val = totals_delta.loc[d]
                mask = (agg['日期'].astype(str) == str(d)) & (agg['指数名称'].astype(str) == '同花顺全A(沪深京)')
                if not any(mask):
                    new_row = {
                        '日期': d,
                        '指数代码': '883957.TI',
                        '指数名称': '同花顺全A(沪深京)',
                        '成交额(元)': total_val,
                        '涨跌幅(%)': np.nan,
                        '主力净流入(元)': np.nan,
                        '较上一交易日成交额(元)': delta_val,
                        '近三交易日成交额(元)': roll_val,
                    }
                    agg = pd.concat([agg, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    idxs = agg.index[mask]
                    agg.loc[idxs, '成交额(元)'] = agg.loc[idxs, '成交额(元)'].where(pd.notna(agg.loc[idxs, '成交额(元)']), total_val)
                    agg.loc[idxs, '较上一交易日成交额(元)'] = agg.loc[idxs, '较上一交易日成交额(元)'].where(pd.notna(agg.loc[idxs, '较上一交易日成交额(元)']), delta_val)
                    agg.loc[idxs, '近三交易日成交额(元)'] = agg.loc[idxs, '近三交易日成交额(元)'].where(pd.notna(agg.loc[idxs, '近三交易日成交额(元)']), roll_val)
        except Exception:
            pass
        try:
            agg = agg.sort_values(['日期','指数名称'])
            agg = agg.drop_duplicates(subset=['日期','指数名称'], keep='last')
        except Exception:
            pass
        # enrich names from probe files if available
        try:
            for d in dates:
                p_probe = os.path.join(os.path.dirname(__file__), f'probe_zhishu_{d}.csv')
                if os.path.exists(p_probe):
                    probe = pd.read_csv(p_probe, encoding='utf-8-sig')
                    probe_cols = [str(c).strip() for c in probe.columns]
                    code_col = next((c for c in probe_cols if ('指数代码' in c)), None)
                    short_col = next((c for c in probe_cols if ('指数简称' in c)), None)
                    name_col2 = next((c for c in probe_cols if ('指数名称' in c)), None)
                    if code_col is not None and (short_col is not None or name_col2 is not None):
                        probe_map = {}
                        for _, r in probe.iterrows():
                            k = str(r[code_col])
                            v = str(r[short_col]) if short_col is not None else (str(r[name_col2]) if name_col2 is not None else '')
                            if k and v:
                                probe_map[k] = v
                        mask_date = agg['日期'].astype(str) == str(d)
                        mask_need = mask_date & ((agg['指数名称'].isna()) | (agg['指数名称'].astype(str) == agg['指数代码'].astype(str)) | (agg['指数名称'].astype(str).str.match(r"^[A-Z0-9\._]+$")))
                        if any(mask_need):
                            def map_name(code, old):
                                return probe_map.get(str(code), old)
                            idxs = agg.index[mask_need]
                            agg.loc[idxs, '指数名称'] = [map_name(agg.loc[i, '指数代码'], agg.loc[i, '指数名称']) for i in idxs]
                    # fill numeric metrics from probe for date d
                    amt_col = next((c for c in probe_cols if (('成交额' in c) and (str(d) in c))), None)
                    pct_col = next((c for c in probe_cols if (('涨跌幅' in c) and (str(d) in c))), None)
                    if code_col is not None and (amt_col is not None or pct_col is not None):
                        mp_amt = {}
                        mp_pct = {}
                        for _, r in probe.iterrows():
                            k = str(r[code_col])
                            if amt_col is not None:
                                try:
                                    v_amt = pd.to_numeric(str(r[amt_col]).replace(',',''), errors='coerce')
                                except Exception:
                                    v_amt = np.nan
                                mp_amt[k] = v_amt
                            if pct_col is not None:
                                try:
                                    v_pct = pd.to_numeric(str(r[pct_col]).replace('%',''), errors='coerce')
                                except Exception:
                                    v_pct = np.nan
                                mp_pct[k] = v_pct
                        mask_d = agg['日期'].astype(str) == str(d)
                        idxs_d = agg.index[mask_d]
                        for i in idxs_d:
                            code_i = str(agg.loc[i, '指数代码'])
                            if pd.isna(agg.loc[i, '成交额(元)']) and (code_i in mp_amt):
                                agg.loc[i, '成交额(元)'] = mp_amt[code_i]
                            if pd.isna(agg.loc[i, '涨跌幅(%)']) and (code_i in mp_pct):
                                agg.loc[i, '涨跌幅(%)'] = mp_pct[code_i]
        except Exception:
            pass
        if '近三日大单净额(元)' not in agg.columns:
            agg['近三日大单净额(元)'] = np.nan
        agg = agg[['日期','指数代码','指数名称','成交额(元)','涨跌幅(%)','主力净流入(元)','较上一交易日成交额(元)','近三交易日成交额(元)','近三日大单净额(元)']]
        agg.to_csv(board_csv_path(), index=False, encoding='utf-8-sig')
        return True
    return False

def get_cached_dates(limit: int = 3):
    try:
        p_all = all_market_cache_path()
        if os.path.exists(p_all):
            hist = pd.read_csv(p_all, encoding='utf-8-sig')
            if '日期' in hist.columns:
                ds = sorted([str(x) for x in hist['日期'].astype(str).unique()])
                return ds[-min(limit, len(ds)):] if limit else ds
    except Exception:
        pass
    return []

def probe_zhishu_store(date):
    try:
        r = pywencai.get(query='当日成交额,较上一交易日成交额,近三交易日成交额', query_type='zhishu', loop=True)
        df0 = None
        if isinstance(r, pd.DataFrame) and not r.empty:
            df0 = r
        elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
            df0 = r[0]
        elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
            df0 = r['tableV1']
        if not isinstance(df0, pd.DataFrame) or df0.empty:
            return False
        df = df0.copy()
        df.columns = [re.sub(pattern, '', str(c)).strip() for c in df.columns]
        code_col = next((c for c in df.columns if ('指数代码' in str(c))), None)
        name_col = next((c for c in df.columns if ('指数简称' in str(c)) or ('指数名称' in str(c)) or ('指数' in str(c))), None)
        amt_today_col = next((c for c in df.columns if ('当日' in str(c) and '成交额' in str(c))), None)
        amt_delta_col = next((c for c in df.columns if ('较上' in str(c) and '成交额' in str(c))), None)
        amt_3day_col = next((c for c in df.columns if ('近三' in str(c) and '成交额' in str(c))), None)
        def parse_amt(val, colname):
            s = str(val)
            mul = 1.0
            cn = str(colname)
            if ('亿' in cn) or ('亿元' in cn) or ('亿' in s):
                mul = 1e8
            elif ('万' in cn) or ('万元' in cn) or ('万' in s):
                mul = 1e4
            ts = re.sub(r"[^0-9\.-]", '', s)
            v = pd.to_numeric(ts, errors='coerce')
            return v*mul if pd.notna(v) else np.nan
        def series_parse(col):
            if col is None:
                return pd.Series([np.nan]*len(df))
            return df[col].apply(lambda x: parse_amt(x, col))
        idx_name = df[name_col].astype(str) if name_col is not None else (df[code_col].astype(str) if code_col is not None else pd.Series(['']*len(df)))
        out = pd.DataFrame({
            '日期': [date]*len(df),
            '指数代码': df[code_col].astype(str) if code_col is not None else '',
            '指数名称': idx_name,
            '成交额(元)': series_parse(amt_today_col),
            '涨跌幅(%)': pd.Series([np.nan]*len(df)),
            '主力净流入(元)': pd.Series([np.nan]*len(df)),
            '较上一交易日成交额(元)': series_parse(amt_delta_col),
            '近三交易日成交额(元)': series_parse(amt_3day_col),
        })
        try:
            cnt = pd.to_numeric(out['成交额(元)'], errors='coerce').notna().sum()
            sample = out[['指数名称','成交额(元)']].head(3).to_dict(orient='records')
            print(f"探测三字段: 成交额有效行数={cnt}, 示例={sample}")
        except Exception:
            pass
        if os.path.exists(board_csv_path()):
            base = pd.read_csv(board_csv_path(), encoding='utf-8-sig')
            base = base[base['日期'].astype(str) != str(date)]
            out = pd.concat([base, out], ignore_index=True)
        out = out[['日期','指数代码','指数名称','成交额(元)','涨跌幅(%)','主力净流入(元)','较上一交易日成交额(元)','近三交易日成交额(元)']]
        out = out.drop_duplicates(subset=['日期','指数名称'], keep='last')
        out.to_csv(board_csv_path(), index=False, encoding='utf-8-sig')
        return True
    except Exception:
        return False

def ingest_probe_zhishu(date):
    try:
        p = os.path.join(os.path.dirname(__file__), f'probe_zhishu_{date}.csv')
        if not os.path.exists(p):
            return False
        df = pd.read_csv(p, encoding='utf-8-sig')
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cols = df.columns.tolist()
        code_col = next((c for c in cols if ('指数代码' in c)), None)
        name_col = next((c for c in cols if ('指数简称' in c)), None)
        if name_col is None:
            name_col = next((c for c in cols if ('指数名称' in c)), None)
        if name_col is None:
            name_col = next((c for c in cols if ('指数' in c and ('指数代码' not in c) and ('指数@' not in c))), None)
        # 精确匹配“指数@成交额[YYYYMMDD]”及日期变体
        try:
            d_dot = datetime.strptime(date, '%Y%m%d').strftime('%Y.%m.%d')
            d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        except Exception:
            d_dot = date
            d_dash = date
        amt_today_col = next((c for c in cols if ('成交额' in c) and ((str(date) in c) or (d_dot in c) or (d_dash in c))), None)
        try:
            d_prev = (datetime.strptime(date, '%Y%m%d') - pd.Timedelta(days=1)).strftime('%Y%m%d')
        except Exception:
            d_prev = None
        # 前一日成交额列
        try:
            d_prev_dot = datetime.strptime(d_prev, '%Y%m%d').strftime('%Y.%m.%d') if d_prev else None
            d_prev_dash = datetime.strptime(d_prev, '%Y%m%d').strftime('%Y-%m-%d') if d_prev else None
        except Exception:
            d_prev_dot = d_prev
            d_prev_dash = d_prev
        amt_prev_col = next((c for c in cols if ('成交额' in c) and (d_prev and ((str(d_prev) in c) or (d_prev_dot and (d_prev_dot in c)) or (d_prev_dash and (d_prev_dash in c))))), None)
        prev_delta_col = next((c for c in cols if ('较上一交易日' in c) and ('成交额' in c)), None)
        pct_col = next((c for c in cols if ('涨跌幅' in c) and (str(date) in c)), None)
        # 近三日大单净额（区间）
        big3_col = next((c for c in cols if ('区间dde大单净额' in c)), None)
        # 主力净流入/资金流向列
        inflow_col = next((c for c in cols if (('主力' in c) and (('净流入' in c) or ('资金流向' in c)))), None)
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
            return v*mul if pd.notna(v) else np.nan
        rows = []
        for _, r in df.iterrows():
            code = str(r[code_col]) if code_col else ''
            name = str(r[name_col]) if name_col else ''
            amt_today = pd.to_numeric(str(r[amt_today_col]).replace(',',''), errors='coerce') if amt_today_col else np.nan
            amt_prev = pd.to_numeric(str(r[amt_prev_col]).replace(',',''), errors='coerce') if amt_prev_col else np.nan
            pct = pd.to_numeric(str(r[pct_col]).replace('%',''), errors='coerce') if pct_col else np.nan
            delta = pd.to_numeric(str(r[prev_delta_col]).replace(',',''), errors='coerce') if prev_delta_col else (amt_today - amt_prev if (pd.notna(amt_today) and pd.notna(amt_prev)) else np.nan)
            big3 = pd.to_numeric(str(r[big3_col]).replace(',',''), errors='coerce') if big3_col else np.nan
            inflow = parse_amt_unit(r[inflow_col]) if inflow_col else np.nan
            rows.append({'日期': str(date), '指数代码': code, '指数名称': name, '成交额(元)': amt_today, '涨跌幅(%)': pct, '主力净流入(元)': inflow, '较上一交易日成交额(元)': delta, '近三交易日成交额(元)': np.nan, '近三日大单净额(元)': big3})
            # add previous day row when available to enable diff/rolling
            if pd.notna(amt_prev):
                rows.append({'日期': str(d_prev) if d_prev else '', '指数代码': code, '指数名称': name, '成交额(元)': amt_prev, '涨跌幅(%)': np.nan, '主力净流入(元)': np.nan, '较上一交易日成交额(元)': np.nan, '近三交易日成交额(元)': np.nan, '近三日大单净额(元)': np.nan})
        out = pd.DataFrame(rows)
        try:
            base = pd.read_csv(board_csv_path(), encoding='utf-8-sig') if os.path.exists(board_csv_path()) else pd.DataFrame()
        except Exception:
            base = pd.DataFrame()
        if not base.empty:
            base = base[base['日期'].astype(str) != str(date)]
            out = pd.concat([base, out], ignore_index=True)
        # ensure new column exists
        if '近三日大单净额(元)' not in out.columns:
            out['近三日大单净额(元)'] = np.nan
        out = out[['日期','指数代码','指数名称','成交额(元)','涨跌幅(%)','主力净流入(元)','较上一交易日成交额(元)','近三交易日成交额(元)','近三日大单净额(元)']]
        if '指数代码' in out.columns:
            out = out.drop_duplicates(subset=['日期','指数代码'], keep='last')
        else:
            out = out.drop_duplicates(subset=['日期','指数名称'], keep='last')
        out.to_csv(board_csv_path(), index=False, encoding='utf-8-sig')
        return True
    except Exception:
        return False

def update_board_from_params(date):
    try:
        try:
            d_dot = datetime.strptime(date, '%Y%m%d').strftime('%Y.%m.%d')
            d_dash = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            d_prev = (datetime.strptime(date, '%Y%m%d') - pd.Timedelta(days=1)).strftime('%Y%m%d')
            d_prev_dot = datetime.strptime(d_prev, '%Y%m%d').strftime('%Y.%m.%d')
            d_prev_dash = datetime.strptime(d_prev, '%Y%m%d').strftime('%Y-%m-%d')
        except Exception:
            d_dot = date
            d_dash = date
            d_prev = None
            d_prev_dot = None
            d_prev_dash = None
        q = "同花顺概念指数 成交额,上1交易日的成交额,板块资金净额,涨跌幅,涨停家数"
        r = pywencai.get(query=q, query_type='zhishu', loop=True)
        df = None
        if isinstance(r, pd.DataFrame) and not r.empty:
            df = r
        elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
            df = r[0]
        elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
            df = r['tableV1']
        if (not isinstance(df, pd.DataFrame)) or df.empty:
            q_alt = "同花顺概念指数 成交额(元), 上1交易日的成交额(元), 资金流向(元), 涨停家数, 涨跌幅(%)"
            r2 = pywencai.get(query=q_alt, query_type='zhishu', loop=True)
            if isinstance(r2, pd.DataFrame) and not r2.empty:
                df = r2
            elif isinstance(r2, list) and len(r2) > 0 and isinstance(r2[0], pd.DataFrame):
                df = r2[0]
            elif isinstance(r2, dict) and 'tableV1' in r2 and isinstance(r2['tableV1'], pd.DataFrame):
                df = r2['tableV1']
            if (not isinstance(df, pd.DataFrame)) or df.empty:
                return False
        df_raw = df.copy()
        raw_cols = [str(c) for c in df_raw.columns]
        df = df.copy()
        df.columns = [re.sub(pattern, '', str(c)).strip() for c in df.columns]
        repls = [d_dot, d_dash, d_prev_dot, d_prev_dash]
        df.columns = [reduce(lambda s,t: s.replace(t if t else '', ''), repls, str(c)).strip() for c in df.columns]
        cols = [str(c) for c in df.columns]
        def find_in(candidates, tokens):
            for c in candidates:
                cc = str(c)
                if all(t in cc for t in tokens):
                    return c
            return None
        code_col = find_in(raw_cols, ['指数代码']) or find_in(raw_cols, ['code']) or find_in(raw_cols, ['market_code']) or find_in(cols, ['指数代码'])
        name_col = find_in(raw_cols, ['指数简称']) or find_in(raw_cols, ['指数名称']) or find_in(raw_cols, ['指数']) or find_in(cols, ['指数简称']) or find_in(cols, ['指数名称'])
        amt_today_col = find_in(raw_cols, ['成交额']) or find_in(cols, ['成交额'])
        amt_prev_col = find_in(raw_cols, ['上1', '成交额']) or find_in(raw_cols, ['上一', '成交额']) or find_in(cols, ['上1', '成交额']) or find_in(cols, ['上一', '成交额'])
        inflow_col = find_in(raw_cols, ['资金流向']) or find_in(raw_cols, ['资金流向(元)']) or find_in(raw_cols, ['板块资金净额']) or find_in(raw_cols, ['资金净额']) or find_in(cols, ['资金流向']) or find_in(cols, ['板块资金净额']) or find_in(cols, ['资金净额'])
        pct_col = find_in(raw_cols, ['涨跌幅']) or find_in(cols, ['涨跌幅'])
        zt_col = find_in(raw_cols, ['涨停家数']) or find_in(cols, ['涨停家数'])
        def parse_amt_unit(val):
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
            return v*mul if pd.notna(v) else np.nan
        rows = []
        filt = df
        if code_col is not None:
            try:
                filt = df[df[code_col].astype(str).str.contains('\.TI|\.CSI|\.SZ|\.SH')]
            except Exception:
                filt = df
        for _, r0 in filt.iterrows():
            code = str(r0[code_col]) if code_col else ''
            name = str(r0[name_col]) if name_col else ''
            amt_today = parse_amt_unit(r0[amt_today_col]) if amt_today_col else np.nan
            amt_prev = parse_amt_unit(r0[amt_prev_col]) if amt_prev_col else np.nan
            delta = (amt_today - amt_prev) if (pd.notna(amt_today) and pd.notna(amt_prev)) else np.nan
            inflow = parse_amt_unit(r0[inflow_col]) if inflow_col else np.nan
            pct = pd.to_numeric(str(r0[pct_col]).replace('%',''), errors='coerce') if pct_col else np.nan
            zt = pd.to_numeric(str(r0[zt_col]), errors='coerce') if zt_col else np.nan
            rows.append({'日期': str(date), '指数代码': code, '指数名称': name, '成交额(元)': amt_today, '上1交易日成交额(元)': amt_prev, '资金流向(元)': inflow, '涨停家数(家)': zt, '涨跌幅(%)': pct})
        out = pd.DataFrame(rows)
        if out.empty:
            return False
        # 补采：若关键列缺失，增发一次查询补齐并合并
        try:
            need_fix = out['资金流向(元)'].isna().sum() > 0 or out['上1交易日成交额(元)'].isna().sum() > 0
        except Exception:
            need_fix = True
        if need_fix:
            try:
                q_fix = "同花顺概念指数 上1交易日的成交额(元), 资金流向(元), 涨停家数"
                r_fix = pywencai.get(query=q_fix, query_type='zhishu', loop=True)
                dfix = None
                if isinstance(r_fix, pd.DataFrame) and not r_fix.empty:
                    dfix = r_fix
                elif isinstance(r_fix, list) and len(r_fix) > 0 and isinstance(r_fix[0], pd.DataFrame):
                    dfix = r_fix[0]
                elif isinstance(r_fix, dict) and 'tableV1' in r_fix and isinstance(r_fix['tableV1'], pd.DataFrame):
                    dfix = r_fix['tableV1']
                if isinstance(dfix, pd.DataFrame) and not dfix.empty:
                    dfix = dfix.copy()
                    dfix.columns = [re.sub(pattern, '', str(c)).strip() for c in dfix.columns]
                    repls2 = [d_dot, d_dash, d_prev_dot, d_prev_dash]
                    dfix.columns = [reduce(lambda s,t: s.replace(t if t else '', ''), repls2, str(c)).strip() for c in dfix.columns]
                    # 识别列
                    rc = [str(c) for c in dfix.columns]
                    code_fix = next((c for c in rc if '指数代码' in c), None)
                    name_fix = next((c for c in rc if ('指数简称' in c or '指数名称' in c or c == '指数')), None)
                    prev_fix = next((c for c in rc if ('上1' in c and '成交额' in c) or ('上一' in c and '成交额' in c)), None)
                    inflow_fix = next((c for c in rc if ('资金流向' in c) or ('资金流向(元)' in c)), None)
                    zt_fix = next((c for c in rc if '涨停家数' in c), None)
                    dfix_small = dfix[[x for x in [code_fix, name_fix, prev_fix, inflow_fix, zt_fix] if x]].copy()
                    dfix_small.rename(columns={code_fix:'指数代码', name_fix:'指数名称', prev_fix:'上1交易日成交额(元)_fix', inflow_fix:'资金流向(元)_fix', zt_fix:'涨停家数(家)_fix'}, inplace=True)
                    # 合并
                    key = '指数代码' if '指数代码' in out.columns and '指数代码' in dfix_small.columns else '指数名称'
                    out = out.merge(dfix_small, on=key, how='left')
                    # 补值
                    if '资金流向(元)_fix' in out.columns:
                        out['资金流向(元)'] = out['资金流向(元)'].fillna(out['资金流向(元)_fix'].apply(parse_amt_unit))
                    if '上1交易日成交额(元)_fix' in out.columns:
                        out['上1交易日成交额(元)'] = out['上1交易日成交额(元)'].fillna(out['上1交易日成交额(元)_fix'].apply(parse_amt_unit))
                    if '涨停家数(家)_fix' in out.columns and '涨停家数(家)' in out.columns:
                        out['涨停家数(家)'] = out['涨停家数(家)'].fillna(pd.to_numeric(out['涨停家数(家)_fix'], errors='coerce'))
                    # 清理临时列
                    for c in ['资金流向(元)_fix','上1交易日成交额(元)_fix','涨停家数(家)_fix']:
                        if c in out.columns:
                            out.drop(columns=[c], inplace=True)
            except Exception:
                pass
        try:
            base = pd.read_csv(board_csv_path(), encoding='utf-8-sig') if os.path.exists(board_csv_path()) else pd.DataFrame()
        except Exception:
            base = pd.DataFrame()
        if not base.empty:
            base = base[base['日期'].astype(str) != str(date)]
            out = pd.concat([base, out], ignore_index=True)
        out = out[['日期','指数代码','指数名称','成交额(元)','上1交易日成交额(元)','资金流向(元)','涨停家数(家)','涨跌幅(%)']]
        if '指数代码' in out.columns:
            out = out.drop_duplicates(subset=['日期','指数代码'], keep='last')
        else:
            out = out.drop_duplicates(subset=['日期','指数名称'], keep='last')
        out.to_csv(board_csv_path(), index=False, encoding='utf-8-sig')
        return True
    except Exception:
        return False

def enhance_stock_data(df, market_data, date):
    """增强股票数据分析"""
    # 添加大盘情绪数据
    if market_data is not None and not market_data.empty:
        market_row = market_data.iloc[0]
        df['市场涨停家数'] = market_row.get('涨停家数', len(df))
        df['市场跌停家数'] = market_row.get('跌停家数', 0)
        df['市场上涨家数'] = market_row.get('上涨家数', 0)
        df['市场下跌家数'] = market_row.get('下跌家数', 0)
        df['上证涨跌幅'] = market_row.get('上证指数涨跌幅', 0)
        df['深证涨跌幅'] = market_row.get('深证成指涨跌幅', 0)
        df['创业板涨跌幅'] = market_row.get('创业板指涨跌幅', 0)
    
    # 计算市场情绪分数
    df = calculate_market_sentiment_score(df)
    
    # 概念热度分析
    df = analyze_concept_hotness(df)
    
    # 技术指标增强
    df = enhance_technical_indicators(df)
    df = normalize_money_metrics(df)
    
    # 风险评级
    df = add_risk_rating(df)
    
    # 综合评分
    df = calculate_comprehensive_score(df)
    
    return df

def calculate_market_sentiment_score(df):
    """计算市场情绪分数"""
    if '市场上涨家数' in df.columns and '市场下跌家数' in df.columns:
        total_stocks = df['市场上涨家数'] + df['市场下跌家数']
        up_ratio = df['市场上涨家数'] / (total_stocks + 1e-5)
        
        # 情绪分数计算
        df['市场情绪分数'] = (
            up_ratio * 0.4 + 
            (df['市场涨停家数'] / 100).clip(0, 1) * 0.3 +
            (1 - (df['市场跌停家数'] / 50).clip(0, 1)) * 0.3
        ) * 100
    
    return df

def analyze_concept_hotness(df):
    """分析概念热度"""
    if '所属概念' in df.columns:
        all_concepts = []
        for concepts in df['所属概念'].dropna():
            if isinstance(concepts, str):
                parts = [c.strip() for c in concepts.split(';')]
                parts = [p for p in parts if p and (p not in BLACKLIST_CONCEPTS)]
                all_concepts.extend(parts)
        concept_counts = pd.Series(all_concepts).value_counts() if len(all_concepts) > 0 else pd.Series(dtype=int)
        date_val = None
        if '涨停日期' in df.columns and len(df) > 0:
            try:
                date_val = str(df['涨停日期'].iloc[0])
            except Exception:
                date_val = None
        strength_map = {}
        try:
            bpath = board_csv_path()
            if os.path.exists(bpath):
                bdf = pd.read_csv(bpath, encoding='utf-8-sig')
                bdf['日期'] = bdf['日期'].astype(str)
                if date_val is not None:
                    bdf = bdf[bdf['日期'] == str(date_val)]
                if {'指数名称','涨跌幅(%)','较上一交易日成交额(元)'}.issubset(set(bdf.columns)):
                    bdf['涨跌幅(%)'] = pd.to_numeric(bdf['涨跌幅(%)'], errors='coerce')
                    bdf['较上一交易日成交额(元)'] = pd.to_numeric(bdf['较上一交易日成交额(元)'], errors='coerce')
                    for _, r in bdf.iterrows():
                        nm = str(r['指数名称'])
                        pct = r['涨跌幅(%)'] if pd.notna(r['涨跌幅(%)']) else 0.0
                        delta_amt = r['较上一交易日成交额(元)'] if pd.notna(r['较上一交易日成交额(元)']) else 0.0
                        w = max(pct, 0.0)/float(CONCEPT_CFG.get('pct_div', 5.0)) + max(delta_amt/float(CONCEPT_CFG.get('amt_scale', 1e9)), 0.0)/float(CONCEPT_CFG.get('amt_div', 50.0))
                        strength_map[nm] = w
        except Exception:
            strength_map = {}
        scores = {}
        for k, v in concept_counts.items():
            w = strength_map.get(k, 0.0)
            scores[k] = float(v) * (1.0 + w)
        hot_min = float(CONCEPT_CFG.get('hot_score_min', 3.0))
        hot_topn = int(CONCEPT_CFG.get('hot_topn', 20))
        hot_concepts = [k for k, s in sorted(scores.items(), key=lambda x: x[1], reverse=True) if s >= hot_min][:hot_topn]
        def count_hot_concepts(concept_str):
            if pd.isna(concept_str):
                return 0
            concepts = [c.strip() for c in str(concept_str).split(';')]
            concepts = [p for p in concepts if p and (p not in BLACKLIST_CONCEPTS)]
            return len(set(concepts) & set(hot_concepts))
        df['热门概念数量'] = df['所属概念'].apply(count_hot_concepts)
        df['是否主流热点'] = df['热门概念数量'] >= 1
        def count_all_concepts(concept_str):
            if pd.isna(concept_str):
                return 0
            return len([c.strip() for c in str(concept_str).split(';') if c.strip() and (c.strip() not in BLACKLIST_CONCEPTS)])
        df['所属概念数量'] = df['所属概念'].apply(count_all_concepts)
        if len(hot_concepts) > 0:
            df['当日热门概念'] = ', '.join(hot_concepts[:5])
    
    return df

def enhance_technical_indicators(df):
    """增强技术指标分析"""
    # 量比分级
    if '量比' in df.columns:
        df['量比强度'] = pd.cut(df['量比'], 
                              bins=[0, 0.8, 1.2, 2, 5, float('inf')],
                              labels=['极低', '偏低', '正常', '活跃', '异常活跃'],
                              include_lowest=True)
    
    # 换手率分级
    if '换手率' in df.columns:
        df['换手强度'] = pd.cut(df['换手率'],
                              bins=[0, 3, 10, 20, 30, float('inf')],
                              labels=['低迷', '温和', '活跃', '激烈', '异常'],
                              include_lowest=True)
    
    # 市值分组
    if 'a股市值(不含限售股)' in df.columns:
        df['市值分组'] = pd.cut(df['a股市值(不含限售股)'],
                              bins=[0, 30e8, 100e8, 300e8, 1000e8, float('inf')],
                              labels=['微盘', '小盘', '中盘', '大盘', '超大盘'],
                              include_lowest=True)
    
    # 封单强度
    if '涨停封单量占成交量比' in df.columns:
        df['封单强度'] = pd.cut(df['涨停封单量占成交量比'],
                              bins=[0, 1, 5, 10, 20, float('inf')],
                              labels=['极弱', '较弱', '一般', '较强', '极强'],
                              include_lowest=True)
    
    return df

def normalize_money_metrics(df):
    grp = None
    if '市值分组' in df.columns:
        grp = df['市值分组']
    else:
        if 'a股市值(不含限售股)' in df.columns:
            df['市值分组'] = pd.cut(df['a股市值(不含限售股)'], bins=[0, 30e8, 100e8, 300e8, 1000e8, float('inf')], labels=['微盘','小盘','中盘','大盘','超大盘'], include_lowest=True)
            grp = df['市值分组']
    metrics = ['最新dde大单净额','涨停封单额','涨停封单量占成交量比']
    for m in metrics:
        if m in df.columns and grp is not None:
            s = pd.to_numeric(df[m], errors='coerce')
            z = s.groupby(grp).transform(lambda x: (x - x.mean())/((x.std()) if pd.notna(x.std()) and x.std()!=0 else 1.0))
            df[m + '_z'] = z
    return df

def add_risk_rating(df):
    """添加风险评级"""
    # PE风险提示
    if '市盈率(pe)' in df.columns:
        # 处理无穷大值
        pe_series = df['市盈率(pe)'].replace([np.inf, -np.inf], np.nan)
        df['PE风险等级'] = pd.cut(pe_series,
                                bins=[-np.inf, 0, 30, 60, 100, np.inf],
                                labels=['亏损', '合理', '偏高', '过高', '严重高估'],
                                include_lowest=True)
    
    # 开板风险
    if '涨停开板次数' in df.columns:
        df['封板质量'] = pd.cut(df['涨停开板次数'],
                              bins=[-1, 0, 2, 5, 10, float('inf')],
                              labels=['极好', '良好', '一般', '较差', '极差'],
                              include_lowest=True)
    
    return df

def calculate_comprehensive_score(df):
    """计算综合评分 (0-100分)"""
    scores = []
    w = CONCEPT_CFG.get('score_weights', {'tech':0.30,'money':0.35,'concept':0.25,'board':0.10})
    
    for idx, row in df.iterrows():
        score = 0
        
        # 技术面评分
        tech_score = 0
        if pd.notna(row.get('量比')):
            tech_score += min(row['量比'] / 2 * 15, 15)
        if pd.notna(row.get('换手率')):
            tech_score += min(row['换手率'] / 15 * 10, 10)
        if pd.notna(row.get('振幅')):
            tech_score += min(row['振幅'] / 12 * 5, 5)
        
        # 资金面评分
        money_score = 0
        dde_val = row.get('最新dde大单净额_z') if pd.notna(row.get('最新dde大单净额_z')) else row.get('最新dde大单净额')
        seal_ratio = row.get('涨停封单量占成交量比_z') if pd.notna(row.get('涨停封单量占成交量比_z')) else row.get('涨停封单量占成交量比')
        seal_amt = row.get('涨停封单额_z') if pd.notna(row.get('涨停封单额_z')) else row.get('涨停封单额')
        if pd.notna(dde_val):
            money_score += min(max(float(dde_val), 0), 15)
        if pd.notna(seal_ratio):
            money_score += min(max(float(seal_ratio), 0), 10)
        if pd.notna(seal_amt):
            money_score += min(max(float(seal_amt), 0), 10)
        
        # 概念面评分
        concept_score = 0
        if pd.notna(row.get('热门概念数量')):
            concept_score += min(row['热门概念数量'] * 5, 15)
        if pd.notna(row.get('所属概念数量')):
            concept_score += min(row['所属概念数量'] / 25 * 10, 10)
        
        # 封板质量评分
        board_score = 0
        if pd.notna(row.get('涨停开板次数')):
            board_score += max(10 - row['涨停开板次数'] * 2, 0)
        if pd.notna(row.get('连续涨停天数')):
            board_score += min(row['连续涨停天数'] * 2, 5)
        try:
            ft = row.get('首次涨停时间')
            if isinstance(ft, str) and ':' in ft:
                h, m = ft.split(':')
                ftm = int(h)*60 + int(m)
                if ftm <= 90:
                    board_score += 2
                elif ftm <= 120:
                    board_score += 1
        except Exception:
            pass
        # 拥挤度与分时强度代理（简化版占位）：概念内家数与换手率为代理
        try:
            hotn = pd.to_numeric(row.get('热门概念数量'), errors='coerce')
            hs = pd.to_numeric(row.get('换手率'), errors='coerce')
            if pd.notna(hotn) and pd.notna(hs):
                # 拥挤度过高（热门概念数量大）时轻微降分，分时加强（换手高）时轻微加分
                concept_score += max(-0.5*max(hotn-20, 0), -5)
                tech_score += min(hs/30*2, 2)
        except Exception:
            pass
        
        total_score = tech_score * float(w.get('tech',0.30)) + money_score * float(w.get('money',0.35)) + concept_score * float(w.get('concept',0.25)) + board_score * float(w.get('board',0.10))
        scores.append(min(total_score, 100))
    
    df['综合评分'] = scores
    th = CONCEPT_CFG.get('grade_thresholds', {'D':40,'C':60,'B':75,'A':90,'S':100})
    df['评分等级'] = pd.cut(df['综合评分'],
                return f"{v2:.2f}" if pd.notna(v2) else 'N/A'
            except Exception:
                return 'N/A'
        def fmt_int(v):
            try:
                v2 = pd.to_numeric(v, errors='coerce')
                return f"{int(v2)}" if pd.notna(v2) else 'N/A'
            except Exception:
                return 'N/A'
        def md_table(headers, rows):
            line1 = '| ' + ' | '.join(headers) + ' |'
            line2 = '| ' + ' | '.join(['---']*len(headers)) + ' |'
            lines = [line1, line2]
            for row in rows:
                lines.append('| ' + ' | '.join(row) + ' |')
            return '\n'.join(lines)
        def df_to_rows(dataframe):
            rows = []
            for _, r in dataframe.iterrows():
                row = []
                for c in dataframe.columns:
                    v = r[c]
                    if isinstance(v, (int, np.integer)):
                        row.append(str(int(v)))
                    elif isinstance(v, (float, np.floating)):
                        row.append(f"{float(v):.2f}")
                    else:
                        row.append(str(v))
                rows.append(row)
            return rows
        market_info_table = [
            ['涨停家数', fmt_int(market_row.get('涨停家数'))],
            ['跌停家数', fmt_int(market_row.get('跌停家数'))],
            ['上涨家数', fmt_int(market_row.get('上涨家数'))],
            ['平盘家数', fmt_int(market_row.get('平盘家数'))],
            ['下跌家数', fmt_int(market_row.get('下跌家数'))],
            ['上证涨跌幅(%)', fmt_pct(market_row.get('上证指数涨跌幅'))],
            ['深证涨跌幅(%)', fmt_pct(market_row.get('深证成指涨跌幅'))],
            ['创业板涨跌幅(%)', fmt_pct(market_row.get('创业板指涨跌幅'))],
            ['A股总成交额(亿)', fmt_pct(pd.to_numeric(market_row.get('A股总成交额(元)'), errors='coerce')/1e8 if pd.notna(market_row.get('A股总成交额(元)')) else np.nan)],
            ['情绪分数', fmt_pct(market_row.get('情绪分数'))],
        ]
    else:
        market_info_table = [['数据获取', '失败']]

    trend_info = ""
    try:
        # 确保统一文件存在
        if not os.path.exists(all_market_cache_path()):
            rebuild_all_cache_from_files()
        hist = pd.read_csv(all_market_cache_path(), encoding='utf-8-sig')
        if '日期' in hist.columns:
            # 仅保留有效行
            cols_needed = ['日期','涨停家数','上涨家数','下跌家数','平盘家数','上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅']
            for c in cols_needed:
                if c not in hist.columns:
                    hist[c] = np.nan
            hist = hist[cols_needed].dropna(how='all')
            if not hist.empty:
                hist = hist.sort_values('日期')
                last = hist.tail(min(5, len(hist)))
                # 计算环比变化与MA5
                histn = last.copy()
                for col in ['涨停家数','上涨家数','下跌家数','平盘家数','上证指数涨跌幅','深证成指涨跌幅','创业板指涨跌幅']:
                    histn[col] = pd.to_numeric(histn[col], errors='coerce')
                if '情绪分数' in hist.columns:
                    histn['情绪分数'] = pd.to_numeric(hist['情绪分数'].tail(len(last)), errors='coerce').values
                histn['涨停Δ'] = histn['涨停家数'].diff()
                histn['上涨Δ'] = histn['上涨家数'].diff()
                histn['下跌Δ'] = histn['下跌家数'].diff()
                histn['平盘Δ'] = histn['平盘家数'].diff()
                histn['上证Δ'] = histn['上证指数涨跌幅'].diff()
                histn['深证Δ'] = histn['深证成指涨跌幅'].diff()
                histn['创业Δ'] = histn['创业板指涨跌幅'].diff()
                if '情绪分数' in histn.columns:
                    histn['情绪MA5'] = pd.Series(histn['情绪分数']).rolling(window=5, min_periods=1).mean()
                trend_headers = ['日期','涨停家数','涨停Δ','情绪分数','情绪MA5','上证(%)','上证Δ','深证(%)','深证Δ','创业板(%)','创业Δ','上涨家数','上涨Δ','下跌家数','下跌Δ','平盘家数','平盘Δ']
                trend_rows = []
                for _, r in histn.iterrows():
                    def fmt(v, is_pct=False):
                        if pd.isna(v):
                            return ''
                        return f"{v:.2f}" if is_pct or isinstance(v, (float, np.floating)) else str(int(v))
                    trend_rows.append([
                        str(r['日期']),
                        fmt(r['涨停家数']), fmt(r['涨停Δ']),
                        fmt(r.get('情绪分数')), fmt(r.get('情绪MA5')),
                        fmt(r['上证指数涨跌幅'], True), fmt(r['上证Δ'], True),
                        fmt(r['深证成指涨跌幅'], True), fmt(r['深证Δ'], True),
                        fmt(r['创业板指涨跌幅'], True), fmt(r['创业Δ'], True),
                        fmt(r['上涨家数']), fmt(r['上涨Δ']),
                        fmt(r['下跌家数']), fmt(r['下跌Δ']),
                        fmt(r['平盘家数']), fmt(r['平盘Δ']),
                    ])
                trend_info = md_table(trend_headers, trend_rows)
    except Exception:
        pass
    
    # 龙头分析
    top_stocks_df = df.nlargest(5, '综合评分')[['股票简称', '几天几板', '综合评分', '评分等级']]
    ml_cols = ['股票简称', '最新dde大单净额'] + (['涨停封单量占成交量比'] if '涨停封单量占成交量比' in df.columns else [])
    money_leaders_df = df.nlargest(3, '最新dde大单净额')[ml_cols]
    
    board_counts = df['几天几板'].dropna().value_counts().sort_index()
    board_headers = ['几天几板', '数量', '股票']
    names_map = {}
    if '股票简称' in df.columns:
        tmp = df[['几天几板','股票简称']].dropna()
        try:
            grp = tmp.groupby('几天几板')['股票简称'].apply(lambda s: '、'.join(sorted(set(s.astype(str).tolist()))))
            names_map = grp.to_dict()
        except Exception:
            names_map = {}
    board_rows = [[str(k), str(v), str(names_map.get(k, ''))] for k, v in board_counts.items()]
    
    # 概念分析
    hot_concepts = df['当日热门概念'].iloc[0] if '当日热门概念' in df.columns else 'N/A'
    hotspot_df = df[df.get('是否主流热点', False) == True]
    hotspot_cols = [c for c in ['股票简称','热门概念数量','所属概念数量','几天几板','综合评分','评分等级'] if c in hotspot_df.columns]
    hotspot_df = hotspot_df[hotspot_cols].nlargest(20, '综合评分') if '综合评分' in hotspot_df.columns else hotspot_df

    def df_to_rows(dataframe):
        rows = []
        for _, r in dataframe.iterrows():
            row = []
            for c in dataframe.columns:
                v = r[c]
                if isinstance(v, (int, np.integer)):
                    row.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    row.append(f"{float(v):.2f}")
                else:
                    row.append(str(v))
            rows.append(row)
        return rows

    def md_table(headers, rows):
        line1 = '| ' + ' | '.join(headers) + ' |'
        line2 = '| ' + ' | '.join(['---']*len(headers)) + ' |'
        lines = [line1, line2]
        for row in rows:
            lines.append('| ' + ' | '.join(row) + ' |')
        return '\n'.join(lines)

    report_parts = []
    report_parts.append(f"# {date} 涨停分析报告")
    report_parts.append("## 市场概况")
    report_parts.append(md_table(['指标','数值'], market_info_table))
    if trend_info:
        report_parts.append("\n## 趋势(近7日)")
        report_parts.append(trend_info)
    report_parts.append("\n## 今日龙头 (综合评分前5)")
    report_parts.append(md_table(list(top_stocks_df.columns), df_to_rows(top_stocks_df)))
    report_parts.append("\n## 资金龙头 (大单净额前3)")
    report_parts.append(md_table(list(money_leaders_df.columns), df_to_rows(money_leaders_df)))
    report_parts.append(f"\n## 概念热点\n{hot_concepts}")
    if not hotspot_df.empty:
        report_parts.append("\n### 热点概念股票")
        report_parts.append(md_table(hotspot_cols, df_to_rows(hotspot_df)))
    report_parts.append("\n## 连板梯队")
    report_parts.append(md_table(board_headers, board_rows))
    try:
        report_parts.append("\n## 涨停概念与行业统计")
        concept_series_up = []
        if '所属概念' in df.columns:
            for s in df['所属概念'].dropna().astype(str).tolist():
                parts = [c.strip() for c in s.split(';') if c.strip()]
                concept_series_up.extend(parts)
        counts_up_concept = pd.Series(concept_series_up).value_counts()
        if len(counts_up_concept) > 0:
            df_up_concepts = counts_up_concept.reset_index().head(15)
            df_up_concepts.columns = ['概念','涨停家数']
            report_parts.append("\n### 概念统计（按涨停家数）")
            report_parts.append(md_table(['概念','涨停家数'], df_to_rows(df_up_concepts)))
        if '所属同花顺行业' in df.columns:
            counts_up_ind = df['所属同花顺行业'].dropna().value_counts().reset_index().head(15)
            counts_up_ind.columns = ['行业','涨停家数']
            report_parts.append("\n### 行业统计（按涨停家数）")
            report_parts.append(md_table(['行业','涨停家数'], df_to_rows(counts_up_ind)))
        cols_up_list = []
        for c in ['股票简称','所属同花顺行业','所属概念']:
            if c in df.columns:
                cols_up_list.append(c)
        if cols_up_list:
            df_up_list = df[cols_up_list].copy()
            if '所属概念' in df_up_list.columns:
                df_up_list['所属概念数量'] = df_up_list['所属概念'].apply(lambda s: len([c.strip() for c in str(s).split(';') if c.strip()]))
                cols_up_list = [c for c in ['股票简称','所属同花顺行业','所属概念数量','所属概念'] if c in df_up_list.columns]
            report_parts.append("\n### 涨停股票列表（含行业与概念）")
            report_parts.append(md_table(cols_up_list, df_to_rows(df_up_list[cols_up_list])))
    except Exception:
        pass
    if isinstance(df_down, pd.DataFrame) and not df_down.empty:
        try:
            report_parts.append("\n## 跌停池")
            try:
                # 概念统计
                concept_series = []
                if '所属概念' in df_down.columns:
                    for s in df_down['所属概念'].dropna().astype(str).tolist():
                        parts = [c.strip() for c in s.split(';') if c.strip()]
                        concept_series.extend(parts)
                concept_counts = pd.Series(concept_series).value_counts()
                if len(concept_counts) > 0:
                    df_concepts = concept_counts.reset_index().head(15)
                    df_concepts.columns = ['概念','跌停家数']
                    report_parts.append("\n### 概念统计（按跌停家数）")
                    report_parts.append(md_table(['概念','跌停家数'], df_to_rows(df_concepts)))
            except Exception:
                pass
            try:
                # 行业统计
                if '所属同花顺行业' in df_down.columns:
                    ind_counts = df_down['所属同花顺行业'].dropna().value_counts().reset_index().head(15)
                    ind_counts.columns = ['行业','跌停家数']
                    report_parts.append("\n### 行业统计（按跌停家数）")
                    report_parts.append(md_table(['行业','跌停家数'], df_to_rows(ind_counts)))
            except Exception:
                pass
            try:
                # 跌停股票列表
                cols_list = [c for c in ['股票简称','所属同花顺行业','所属概念数量','所属概念'] if c in df_down.columns]
                if cols_list:
                    lst = df_down[cols_list]
                    report_parts.append("\n### 跌停股票列表（含行业与概念）")
                    report_parts.append(md_table(cols_list, df_to_rows(lst)))
            except Exception:
                pass
        except Exception:
            pass
    try:
        if os.path.exists(board_csv_path()):
            bdf = pd.read_csv(board_csv_path(), encoding='utf-8-sig')
            if {'日期','指数名称','指数代码','成交额(元)','涨跌幅(%)'}.issubset(set(bdf.columns)):
                bdf['日期'] = bdf['日期'].astype(str)
                uniq = sorted(bdf['日期'].unique())
                recent_dates3 = uniq[-min(3, len(uniq)):]
                recent = bdf[bdf['日期'].isin(recent_dates3)]
                def name_needs_fix(s):
                    try:
                        v = str(s)
                        if v == '' or v.lower() == 'nan':
                            return True
                        if re.match(r"^[A-Z0-9\._]+$", v):
                            return True
                        return False
                    except Exception:
                        return True
                def fill_names(df, d):
                    try:
                        p_probe = os.path.join(os.path.dirname(__file__), f"probe_zhishu_{d}.csv")
                        if not os.path.exists(p_probe):
                            return df
                        probe = pd.read_csv(p_probe, encoding='utf-8-sig')
                        cols = [str(c).strip() for c in probe.columns]
                        code_col = next((c for c in cols if ('指数代码' in c)), None)
                        short_col = next((c for c in cols if ('指数简称' in c)), None)
                        if code_col is None or short_col is None:
                            return df
                        mp = {}
                        for _, r in probe.iterrows():
                            k = str(r[code_col])
                            v = str(r[short_col])
                            if k and v:
                                mp[k] = v
                        mask = (df['日期'].astype(str) == str(d)) & df['指数名称'].apply(name_needs_fix)
                        if any(mask):
                            idxs = df.index[mask]
                            df.loc[idxs, '指数名称'] = [mp.get(str(df.loc[i, '指数代码']), df.loc[i, '指数名称']) for i in idxs]
                    except Exception:
                        return df
                    return df
                for d in recent_dates3:
                    recent = fill_names(recent, d)
                # fallback: build code->name map from existing non-code names and apply globally
                try:
                    valid = recent[~recent['指数名称'].apply(name_needs_fix)]
                    if not valid.empty and {'指数代码','指数名称'}.issubset(set(valid.columns)):
                        code_name_map = {}
                        for _, r in valid.iterrows():
                            k = str(r['指数代码'])
                            v = str(r['指数名称'])
                            if k and v:
                                code_name_map[k] = v
                        mask_fix = recent['指数名称'].apply(name_needs_fix) & recent['指数代码'].astype(str).isin(list(code_name_map.keys()))
                        if any(mask_fix):
                            idxs = recent.index[mask_fix]
                            recent.loc[idxs, '指数名称'] = [code_name_map.get(str(recent.loc[i, '指数代码']), recent.loc[i, '指数名称']) for i in idxs]
                except Exception:
                    pass
                latest_date = sorted(recent['日期'].unique())[-1]
                prev_date = sorted(recent['日期'].unique())[0] if len(sorted(recent['日期'].unique()))>1 else latest_date
                latest = recent[recent['日期']==latest_date].copy()
                prev = recent[recent['日期']==prev_date].copy()
                try:
                    latest = latest.drop_duplicates(subset=['指数名称'], keep='last')
                    prev = prev.drop_duplicates(subset=['指数名称'], keep='last')
                except Exception:
                    pass
                merged = pd.merge(latest, prev[['指数名称','涨跌幅(%)']], on='指数名称', how='left', suffixes=('', '_前日'))
                merged['涨幅Δ(%)'] = merged['涨跌幅(%)'] - merged['涨跌幅(%)_前日']
                merged['成交额(亿)'] = (pd.to_numeric(merged['成交额(元)'], errors='coerce')/1e8).round(2)
                merged['资金流入(亿)'] = (pd.to_numeric(merged['较上一交易日成交额(元)'], errors='coerce')/1e8).round(2)
                merged['主力净流入(亿)'] = (pd.to_numeric(merged['主力净流入(元)'], errors='coerce')/1e8).round(2)
                merged['近三大单净额(亿)'] = (pd.to_numeric(merged['近三日大单净额(元)'], errors='coerce')/1e8).round(2)
                req = ['指数名称','指数代码','涨跌幅(%)','成交额(元)','较上一交易日成交额(元)','主力净流入(元)','近三日大单净额(元)']
                merged_valid = merged.dropna(subset=req).copy()
                merged_valid['涨跌幅(%)'] = pd.to_numeric(merged_valid['涨跌幅(%)'], errors='coerce').fillna(0)
                merged_valid['成交额(元)'] = pd.to_numeric(merged_valid['成交额(元)'], errors='coerce')
                merged_valid['较上一交易日成交额(元)'] = pd.to_numeric(merged_valid['较上一交易日成交额(元)'], errors='coerce')
                merged_valid['近三日大单净额(元)'] = pd.to_numeric(merged_valid['近三日大单净额(元)'], errors='coerce')
                merged_valid['主力净流入(元)'] = pd.to_numeric(merged_valid['主力净流入(元)'], errors='coerce')
                strong = merged_valid.sort_values(['涨跌幅(%)','较上一交易日成交额(元)','近三日大单净额(元)','成交额(元)'], ascending=[False, False, False, False])
                weak = merged_valid.sort_values(['涨跌幅(%)','较上一交易日成交额(元)','近三日大单净额(元)','成交额(元)'], ascending=[True, True, True, True])
                try:
                    strong = strong.drop_duplicates(subset=['指数名称','指数代码'], keep='first').head(10)
                    weak = weak.drop_duplicates(subset=['指数名称','指数代码'], keep='first').head(10)
                except Exception:
                    strong = strong.head(10)
                    weak = weak.head(10)
                strong_headers = ['指数名称','指数代码','涨跌幅(%)','涨幅Δ(%)','成交额(亿)','资金流入(亿)','主力净流入(亿)','近三大单净额(亿)']
                weak_headers = strong_headers
                pass
                pass
    except Exception:
        pass
    try:
        if os.path.exists(board_csv_path()):
            bdf2 = pd.read_csv(board_csv_path(), encoding='utf-8-sig')
            bdf2['日期'] = bdf2['日期'].astype(str)
            latest_date2 = str(date)
            sub2 = bdf2[bdf2['日期'] == latest_date2]
            if not sub2.empty and {'指数名称','涨跌幅(%)','较上一交易日成交额(元)','主力净流入(元)'}.issubset(set(sub2.columns)):
                def nz(x):
                    return pd.to_numeric(x, errors='coerce').fillna(0.0)
                z_pct = (nz(sub2['涨跌幅(%)']) - nz(sub2['涨跌幅(%)']).mean())/((nz(sub2['涨跌幅(%)']).std()) if nz(sub2['涨跌幅(%)']).std()!=0 else 1.0)
                z_delta = (nz(sub2['较上一交易日成交额(元)']) - nz(sub2['较上一交易日成交额(元)']).mean())/((nz(sub2['较上一交易日成交额(元)']).std()) if nz(sub2['较上一交易日成交额(元)']).std()!=0 else 1.0)
                z_main = (nz(sub2['主力净流入(元)']) - nz(sub2['主力净流入(元)']).mean())/((nz(sub2['主力净流入(元)']).std()) if nz(sub2['主力净流入(元)']).std()!=0 else 1.0)
                sub2 = sub2.assign(__score = z_pct*0.5 + z_delta*0.3 + z_main*0.2)
                top_boards = sub2.sort_values('__score', ascending=False)['指数名称'].astype(str).tolist()[:5]
                def in_board(row, name):
                    ok = False
                    if '所属概念' in df.columns:
                        try:
                            ok = name in str(row.get('所属概念',''))
                        except Exception:
                            ok = False
                    if (not ok) and ('所属同花顺行业' in df.columns):
                        try:
                            ok = name in str(row.get('所属同花顺行业',''))
                        except Exception:
                            ok = False
                    return ok
                def parse_boards(s):
                    try:
                        ss = str(s)
                        if '首板' in ss:
                            return 1
                        nums = re.findall(r"(\d+)板", ss)
                        if nums:
                            return int(nums[0])
                    except Exception:
                        return 0
                    return 0
                leaders_rows = []
                followers_rows = []
                for bd in top_boards:
                    group = df[df.apply(lambda r: in_board(r, bd), axis=1)]
                    if group.empty:
                        continue
                    group = group.copy()
                    # 注入归属信息
                    group['归属板块'] = bd
                    def source_type(r):
                        try:
                            if ('所属概念' in df.columns) and (bd in str(r.get('所属概念',''))):
                                return '概念'
                            if ('所属同花顺行业' in df.columns) and (bd in str(r.get('所属同花顺行业',''))):
                                return '行业'
                        except Exception:
                            return ''
                        return ''
                    group['归属类型'] = group.apply(source_type, axis=1)
                    group['连板数'] = group['几天几板'].apply(parse_boards) if '几天几板' in group.columns else 0
                    sort_cols = []
                    if '连板数' in group.columns:
                        sort_cols.append('连板数')
                    if '综合评分' in group.columns:
                        sort_cols.append('综合评分')
                    if '最新dde大单净额_z' in group.columns:
                        sort_cols.append('最新dde大单净额_z')
                    if '涨停封单量占成交量比_z' in group.columns:
                        sort_cols.append('涨停封单量占成交量比_z')
                    group_sorted = group.sort_values(sort_cols, ascending=[False]*len(sort_cols)) if sort_cols else group
                    leader_pool = group_sorted[group_sorted['连板数'] >= 2] if '连板数' in group_sorted.columns else pd.DataFrame()
                    if not leader_pool.empty:
                        leader = leader_pool.iloc[[0]].copy()
                        leader['龙头性质'] = '连板龙头'
                        def reason_row(r):
                            vals = []
                            try:
                                vals.append(f"连板={int(r.get('连板数',0))}")
                            except Exception:
                                pass
                            try:
                                vals.append(f"评分={float(r.get('综合评分',0)):.2f}")
                            except Exception:
                                pass
                            return '，'.join(vals)
                        leader['判定原因'] = leader.apply(reason_row, axis=1)
                        leaders_rows.append(leader)
                    tail = group_sorted.iloc[1:6] if len(group_sorted) > 1 else pd.DataFrame()
                    if not tail.empty:
                        tail = tail.copy()
                        tail['龙头性质'] = tail['几天几板'].apply(lambda s: '跟风') if '几天几板' in tail.columns else '跟风'
                        followers_rows.append(tail)
                if leaders_rows:
                    leaders_df = pd.concat(leaders_rows, ignore_index=True)
                    if '股票简称' in leaders_df.columns:
                        leaders_df = leaders_df.drop_duplicates(subset=['股票简称'], keep='first')
                    cols_ml = [c for c in ['股票简称','几天几板','龙头性质','归属板块','归属类型','判定原因','综合评分','评分等级','热门概念数量'] if c in leaders_df.columns]
                    leaders_df = leaders_df.sort_values(['龙头性质','几天几板','综合评分'], ascending=[True, False, False]) if {'龙头性质','几天几板','综合评分'}.issubset(set(leaders_df.columns)) else leaders_df
                    report_parts.append("\n### 主线龙头")
                    report_parts.append(md_table(cols_ml, df_to_rows(leaders_df[cols_ml])))
                if followers_rows:
                    followers_df = pd.concat(followers_rows, ignore_index=True)
                    if '股票简称' in followers_df.columns:
                        followers_df = followers_df.drop_duplicates(subset=['股票简称'], keep='first')
                    cols_fw = [c for c in ['股票简称','几天几板','归属板块','归属类型','综合评分','评分等级','热门概念数量'] if c in followers_df.columns]
                    followers_df = followers_df.sort_values(['几天几板','综合评分'], ascending=[False, False]) if {'几天几板','综合评分'}.issubset(set(followers_df.columns)) else followers_df
                    report_parts.append("\n### 跟风梯队")
                    report_parts.append(md_table(cols_fw, df_to_rows(followers_df[cols_fw].head(10))))
    except Exception:
        pass
    report_parts.append("\n## 操作建议")
    report_parts.append(md_table(['类别','数量','建议'], [
        ['S级标的', str(len(df[df['评分等级'] == 'S'])), '重点关注'],
        ['A级标的', str(len(df[df['评分等级'] == 'A'])), '适当参与'],
        ['高风险标', str(len(df[df.get('PE风险等级','') == '严重高估'])), '谨慎对待'],
    ]))
    try:
        if market_data is not None and not market_data.empty and ('情绪分数' in market_data.columns):
            emo = pd.to_numeric(market_data.iloc[0].get('情绪分数'), errors='coerce')
            pos = ''
            if pd.notna(emo):
                if emo >= 70:
                    pos = '进攻型'
                elif emo >= 50:
                    pos = '均衡型'
                elif emo >= 30:
                    pos = '防守型'
                else:
                    pos = '观望'
                report_parts.append("\n### 仓位建议分档")
                report_parts.append(md_table(['情绪分数','建议仓位'], [[f"{float(emo):.2f}", pos]]))
    except Exception:
        pass
    try:
        report_parts.append("\n## 板块资金方向（简明版）")
        append_board_net_pct_sections(report_parts, date)
    except Exception:
        pass
    try:
        if '持仓' in df.columns and df['持仓'].any():
            hold_df = df[df['持仓'] == True]
            view_cols = [c for c in ['股票简称','几天几板','综合评分','评分等级','涨停开板次数','最新dde大单净额','换手率','是否主流热点'] if c in hold_df.columns]
            hold_view = hold_df[view_cols]
            s_cnt = int((hold_df['评分等级'] == 'S').sum()) if '评分等级' in hold_df.columns else 0
            a_cnt = int((hold_df['评分等级'] == 'A').sum()) if '评分等级' in hold_df.columns else 0
            risk_cnt = int((hold_df['PE风险等级'] == '严重高估').sum()) if 'PE风险等级' in hold_df.columns else 0
            report_parts.append("\n## 持仓分析")
            report_parts.append(md_table(view_cols, df_to_rows(hold_view)))
            report_parts.append("\n### 持仓建议汇总")
            report_parts.append(md_table(['项','数量','建议'], [["S级持仓", str(s_cnt), "重点持有"], ["A级持仓", str(a_cnt), "观察加仓"], ["高估风险", str(risk_cnt), "控仓或止盈"]]))
            try:
                numfmt = lambda x: (f"{float(x):.2f}" if pd.notna(x) else "")
                dde_sum = pd.to_numeric(hold_df.get('最新dde大单净额'), errors='coerce').sum() if '最新dde大单净额' in hold_df.columns else np.nan
                turn_avg = pd.to_numeric(hold_df.get('换手率'), errors='coerce').mean() if '换手率' in hold_df.columns else np.nan
                pct_avg = pd.to_numeric(hold_df.get('最新涨跌幅'), errors='coerce').mean() if '最新涨跌幅' in hold_df.columns else np.nan
                amt_sum = pd.to_numeric(hold_df.get('成交额'), errors='coerce').sum() if '成交额' in hold_df.columns else np.nan
                report_parts.append("\n### 持仓资金摘要")
                report_parts.append(md_table(['项','数值'], [["最新dde合计(元)", numfmt(dde_sum)], ["平均换手率(%)", numfmt(turn_avg)], ["平均涨跌幅(%)", numfmt(pct_avg)], ["成交额合计(元)", numfmt(amt_sum)]]))
            except Exception:
                pass
            try:
                concepts_series = hold_df.get('所属概念') if '所属概念' in hold_df.columns else None
                if concepts_series is not None:
                    allc = []
                    for s in concepts_series.dropna().astype(str):
                        parts = [p.strip() for p in s.split(';') if p.strip()]
                        parts = [p for p in parts if p and (p not in BLACKLIST_CONCEPTS)]
                        allc.extend(parts)
                    vc = pd.Series(allc).value_counts() if len(allc) > 0 else pd.Series(dtype=int)
                    top = vc.head(15)
                    report_parts.append("\n### 持仓概念Top")
                    report_parts.append(md_table(['概念','持仓家数'], [[str(k), str(int(v))] for k, v in top.items()]))
            except Exception:
                pass
            try:
                if '所属同花顺行业' in hold_df.columns:
                    vc2 = hold_df['所属同花顺行业'].fillna('').astype(str).value_counts()
                    top2 = vc2.head(10)
                    report_parts.append("\n### 持仓行业分布")
                    report_parts.append(md_table(['行业','持仓家数'], [[str(k), str(int(v))] for k, v in top2.items()]))
            except Exception:
                pass
            try:
                prev_date = None
                try:
                    from datetime import datetime, timedelta
                    prev_date = (datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                except Exception:
                    prev_date = None
                prev_names = []
                if prev_date is not None:
                    try:
                        hdf_prev = pd.read_csv(os.path.join(os.path.dirname(__file__), '持仓股票.csv'), encoding='utf-8-sig')
                        if '日期' in hdf_prev.columns and '股票名称' in hdf_prev.columns:
                            prev_names = hdf_prev[hdf_prev['日期'].astype(str) == str(prev_date)]['股票名称'].astype(str).str.strip().tolist()
                    except Exception:
                        prev_names = []
                verify_df = None
                if prev_names:
                    try:
                        verify_df = fetch_hold_data(date, prev_names)
                    except Exception:
                        verify_df = None
                report_parts.append("\n## 隔日验证（昨日持仓今日表现）")
                if isinstance(verify_df, pd.DataFrame) and not verify_df.empty:
                    cols_v = [c for c in ['股票简称','最新涨跌幅','最新价','换手率','所属同花顺行业','最新dde大单净额'] if c in verify_df.columns]
                    report_parts.append(md_table(cols_v, df_to_rows(verify_df[cols_v])))
                else:
                    report_parts.append("昨日持仓为空或验证查询未命中")
            except Exception:
                pass
        else:
            report_parts.append("\n## 持仓分析\n无当日持仓记录或未匹配到持仓标的")
    except Exception:
        report_parts.append("\n## 持仓分析\n生成持仓分析失败")
    _report = '\n'.join(report_parts)
    try:
        i = _report.find('# ')
        if i > 0:
            _report = _report[i:]
    except Exception:
        pass
    return _report

def append_board_net_pct_sections(report_parts, date):
    if not os.path.exists(board_csv_path()):
        return
    bdf = pd.read_csv(board_csv_path(), encoding='utf-8-sig')
    bdf['日期'] = bdf['日期'].astype(str)
    try:
        d8 = re.sub(r'\D', '', str(date))
        if len(d8) != 8:
            d8 = datetime.strptime(str(date), '%Y-%m-%d').strftime('%Y%m%d')
    except Exception:
        d8 = re.sub(r'\D', '', str(date))
    sub = bdf[bdf['日期'].str.replace(r'\D', '', regex=True) == d8]
    req = ['指数名称','指数代码','成交额(元)','资金流向(元)','涨跌幅(%)']
    if not set(req).issubset(set(sub.columns)):
        return
    sub = sub.copy()
    for c in req:
        sub[c] = sub[c].astype(str).str.strip()
    mask = pd.Series(True, index=sub.index)
    for c in req:
        s = sub[c].astype(str)
        mask = mask & (s != '') & (~s.str.lower().eq('nan'))
    sub = sub[mask]
    if ('涨停家数(家)' not in sub.columns) and ('涨停家数' in sub.columns):
        sub['涨停家数(家)'] = sub['涨停家数']
    sub['成交额(亿)'] = (pd.to_numeric(sub['成交额(元)'], errors='coerce')/1e8).round(2)
    sub['资金净额(亿)'] = (pd.to_numeric(sub['资金流向(元)'], errors='coerce')/1e8).round(2)
    net_top = sub.sort_values(['资金流向(元)','成交额(元)'], ascending=[False, False]).head(20)
    pct_top = sub.sort_values(['涨跌幅(%)','成交额(元)'], ascending=[False, False]).head(20)
    net_headers = ['指数名称','指数代码','资金净额(亿)','成交额(亿)','涨跌幅(%)','涨停家数(家)']
    pct_headers = ['指数名称','指数代码','涨跌幅(%)','成交额(亿)','资金净额(亿)','涨停家数(家)']
    report_parts.append("\n### 板块资金净额榜")
    def _fmt2(v):
        try:
            vn = pd.to_numeric(v, errors='coerce')
            return f"{float(vn):.2f}" if pd.notna(vn) else ''
        except Exception:
            return ''
    try:
        rows_net = [[str(r['指数名称']), str(r['指数代码']), _fmt2(r['资金净额(亿)']), _fmt2(r['成交额(亿)']), _fmt2(r['涨跌幅(%)']), str(r.get('涨停家数(家)',''))] for _, r in net_top.iterrows()]
        report_parts.append(md_table(net_headers, rows_net if len(rows_net) > 0 else [['无数据']]))
    except Exception:
        report_parts.append(md_table(net_headers, [['无数据']]))
    report_parts.append("\n### 板块涨幅榜")
    try:
        rows_pct = [[str(r['指数名称']), str(r['指数代码']), _fmt2(r['涨跌幅(%)']), _fmt2(r['成交额(亿)']), _fmt2(r['资金净额(亿)']), str(r.get('涨停家数(家)',''))] for _, r in pct_top.iterrows()]
        report_parts.append(md_table(pct_headers, rows_pct if len(rows_pct) > 0 else [['无数据']]))
    except Exception:
        report_parts.append(md_table(pct_headers, [['无数据']]))
    inter_names = set(net_top['指数名称'].astype(str)) & set(pct_top['指数名称'].astype(str))
    inter_df = sub[sub['指数名称'].astype(str).isin(inter_names)].copy()
    inter_df['综合序'] = inter_df.apply(lambda r: (list(net_top['指数名称'].astype(str)).index(str(r['指数名称'])) if str(r['指数名称']) in list(net_top['指数名称'].astype(str)) else 999) + (list(pct_top['指数名称'].astype(str)).index(str(r['指数名称'])) if str(r['指数名称']) in list(pct_top['指数名称'].astype(str)) else 999), axis=1)
    inter_df = inter_df.sort_values('综合序').head(20)
    inter_headers = ['指数名称','指数代码','涨跌幅(%)','资金净额(亿)','成交额(亿)','涨停家数(家)']
    report_parts.append("\n### 双强板块（资金净额+涨幅均靠前）")
    try:
        rows_inter = [[str(r['指数名称']), str(r['指数代码']), _fmt2(r['涨跌幅(%)']), _fmt2(r['资金净额(亿)']), _fmt2(r['成交额(亿)']), str(r.get('涨停家数(家)',''))] for _, r in inter_df.iterrows()]
        if len(rows_inter) > 0:
            report_parts.append(md_table(inter_headers, rows_inter))
    except Exception:
        pass
    recent_dates = sorted(bdf['日期'].unique())[-3:]
    if len(recent_dates) >= 2:
        tops = []
        for d0 in recent_dates:
            s0 = bdf[bdf['日期'] == d0].copy()
            s0['成交额(亿)'] = (pd.to_numeric(s0['成交额(元)'], errors='coerce')/1e8).round(2)
            s0['资金净额(亿)'] = (pd.to_numeric(s0['资金流向(元)'], errors='coerce')/1e8).round(2)
            n0 = s0.sort_values(['资金流向(元)','成交额(元)'], ascending=[False, False]).head(20)['指数名称'].astype(str)
            p0 = s0.sort_values(['涨跌幅(%)','成交额(元)'], ascending=[False, False]).head(20)['指数名称'].astype(str)
            tops.append(set(n0) & set(p0))
        sustained = tops[0].intersection(*tops[1:]) if len(tops) >= 2 else set()
        if sustained:
            sus_df = sub[sub['指数名称'].astype(str).isin(sustained)]
            report_parts.append("\n### 持续双强（近3日均在双榜）")
            try:
                rows_sus = [[str(r['指数名称']), str(r['指数代码']), _fmt2(r['涨跌幅(%)']), _fmt2(r['资金净额(亿)']), _fmt2(r['成交额(亿)']), str(r.get('涨停家数(家)',''))] for _, r in sus_df.iterrows()]
                if len(rows_sus) > 0:
                    report_parts.append(md_table(inter_headers, rows_sus))
            except Exception:
                pass

def process_date(date):
    market_data = get_market_sentiment(date)
    query = f'{date}涨停股票,量比,市盈率,所属概念,所属同花顺行业,非st股票,几天几板,涨停开板次数'
    try:
        _res = pywencai.get(query=query, loop=True)
    except Exception:
        _res = None
    df = None
    if isinstance(_res, pd.DataFrame):
        df = _res
    elif isinstance(_res, list):
        for item in _res:
            if isinstance(item, pd.DataFrame):
                df = item
                break
    if not (isinstance(df, pd.DataFrame) and not df.empty):
        try:
            _res2 = pywencai.get(query=f'{date}涨停股票', loop=True)
        except Exception:
            _res2 = None
        if isinstance(_res2, pd.DataFrame):
            df = _res2
        elif isinstance(_res2, list):
            for item in _res2:
                if isinstance(item, pd.DataFrame):
                    df = item
                    break
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.columns = [re.sub(pattern, "", col).strip() for col in df.columns]
        df = normalize_columns(df)
        df = coerce_numeric(df, [
            '量比', '换手率', '振幅', '最新dde大单净额',
            '涨停封单量占成交量比', '涨停封单额', '市盈率(pe)',
            '涨停开板次数', '连续涨停天数'
        ])
        if 'a股市值(不含限售股)' in df.columns:
            df['a股市值(不含限售股)'] = df['a股市值(不含限售股)'].apply(parse_market_cap)
        df['涨停日期'] = date
        try:
            hdf = pd.read_csv(os.path.join(os.path.dirname(__file__), '持仓股票.csv'), encoding='utf-8-sig')
            if '日期' in hdf.columns and '股票名称' in hdf.columns and '股票简称' in df.columns:
                names_series = hdf[hdf['日期'].astype(str) == str(date)]['股票名称'].astype(str).str.strip()
                names_series = names_series[names_series != '']
                holds = set(names_series.tolist())
                df['持仓'] = df['股票简称'].astype(str).isin(holds)
                extra = [n for n in holds if n not in set(df['股票简称'].astype(str))]
                if len(extra) > 0:
                    extra_df = fetch_hold_data(date, extra)
                    if isinstance(extra_df, pd.DataFrame) and not extra_df.empty:
                        df = pd.concat([df, extra_df], ignore_index=True, sort=False)
            else:
                df['持仓'] = False
        except Exception:
            df['持仓'] = False
        df = enhance_stock_data(df, market_data, date)
        print('------------------------' + query + '------------------------')
        enhanced_cols = [
            'code', '股票简称', '最新价', '最新涨跌幅', '涨停', '量比', '量比强度',
            '市盈率(pe)', 'PE风险等级', '所属概念', '所属同花顺行业', 
            '几天几板', '连续涨停天数', '所属概念数量', '热门概念数量', '是否主流热点',
            'a股市值(不含限售股)', '市值分组', '最新dde大单净额', 
            '换手率', '换手强度', '振幅', '首次涨停时间', '最终涨停时间',
            '涨停原因类别', '涨停封单量', '涨停封单额', '涨停封单量占成交量比', '封单强度',
            '涨停开板次数', '封板质量', '综合评分', '评分等级',
            '市场涨停家数', '市场跌停家数', '市场情绪分数', '当日热门概念', '涨停日期', '持仓'
        ]
        available_cols = [col for col in enhanced_cols if col in df.columns]
        try:
            df[available_cols].to_csv(out_path(f'增强涨停{date}.csv'), encoding='utf-8-sig', index=False)
            try:
                qd = pywencai.get(query=f'{date}跌停股票,量比,市盈率,所属概念,所属同花顺行业,非st股票,最新dde大单净额,换手率,a股市值(不含限售股)', loop=True)
            except Exception:
                qd = None
            df_down = None
            if isinstance(qd, pd.DataFrame) and not qd.empty:
                df_down = qd
            elif isinstance(qd, list):
                for item in qd:
                    if isinstance(item, pd.DataFrame) and not item.empty:
                        df_down = item
                        break
            if isinstance(df_down, pd.DataFrame) and not df_down.empty:
                df_down.columns = [re.sub(pattern, "", col).strip() for col in df_down.columns]
                df_down = normalize_columns(df_down)
                df_down = coerce_numeric(df_down, [
                    '量比','换手率','振幅','市盈率(pe)','最新dde大单净额'
                ])
                if 'a股市值(不含限售股)' in df_down.columns:
                    df_down['a股市值(不含限售股)'] = df_down['a股市值(不含限售股)'].apply(parse_market_cap)
                try:
                    if '所属概念' in df_down.columns:
                        df_down['所属概念数量'] = df_down['所属概念'].apply(lambda s: len([c.strip() for c in str(s).split(';') if c.strip()]))
                    else:
                        df_down['所属概念数量'] = 0
                except Exception:
                    df_down['所属概念数量'] = 0
                try:
                    def money_score_row(r):
                        lb = pd.to_numeric(r.get('量比'), errors='coerce')
                        hs = pd.to_numeric(r.get('换手率'), errors='coerce')
                        dde = pd.to_numeric(r.get('最新dde大单净额'), errors='coerce')
                        s1 = min(max((-dde) / 5e7, 0), 15) if pd.notna(dde) else 0
                        s2 = min((hs or 0) / 15 * 10, 10) if pd.notna(hs) else 0
                        s3 = min((lb or 0) / 2 * 5, 5) if pd.notna(lb) else 0
                        return float(s1 + s2 + s3)
                    df_down['资金面评分'] = df_down.apply(money_score_row, axis=1)
                except Exception:
                    df_down['资金面评分'] = 0.0
                try:
                    cols_down_save = [c for c in ['code','股票简称','最新涨跌幅','量比','换手率','振幅','所属同花顺行业','所属概念','所属概念数量','最新dde大单净额','资金面评分','a股市值(不含限售股)'] if c in df_down.columns]
                    if cols_down_save:
                        df_down[cols_down_save].to_csv(out_path(f'增强跌停{date}.csv'), encoding='utf-8-sig', index=False)
                except Exception:
                    pass
            report = generate_daily_analysis_report(df, date, market_data, df_down if isinstance(df_down, pd.DataFrame) else None)
            print(report)
            with open(out_path(f'每日分析报告{date}.md'), 'w', encoding='utf-8') as f:
                f.write(report)
            try:
                dest_dir = r'D:\obsidian\股票\历史记录'
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copyfile(out_path(f'每日分析报告{date}.md'), os.path.join(dest_dir, f'每日分析报告{date}.md'))
            except Exception:
                pass
            try:
                df[df['持仓'] == True][available_cols].to_csv(out_path(f'持仓分析{date}.csv'), encoding='utf-8-sig', index=False)
            except Exception:
                pass
        except KeyError as e:
            print(f"处理数据时出错：{e}")
            base_cols = ['code','股票简称','最新价', '最新涨跌幅', '涨停', '量比',
                       '市盈率(pe)', '所属概念', '所属同花顺行业', '几天几板','所属概念数量',
                       'a股市值(不含限售股)', '最新dde大单净额', '换手率', '振幅', 
                       '首次涨停时间', '最终涨停时间', '连续涨停天数', '涨停原因类别',
                       '涨停封单量', '涨停封单额', '涨停封单量占成交量比', '涨停开板次数', '涨停日期']
            available_base_cols = [col for col in base_cols if col in df.columns]
            df[available_base_cols].to_csv(f'涨停{date}.csv', encoding='utf-8-sig')
    else:
        print(f"查询 '{query}' 没有返回数据。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date')
    parser.add_argument('--update-market', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--update-board', action='store_true')
    parser.add_argument('--update-board-params', action='store_true')
    parser.add_argument('--probe-zhishu', action='store_true')
    parser.add_argument('--ingest-probe-zhishu', action='store_true')
    parser.add_argument('--update-market-from-probe', action='store_true')
    parser.add_argument('--update-market-counts', action='store_true')
    parser.add_argument('--update-board-from-cache', action='store_true')
    parser.add_argument('--board-days', type=int)
    parser.add_argument('--dates')
    parser.add_argument('--ingest-market-from-fetch', action='store_true')
    args = parser.parse_args()
    if args.date:
        datelist = [args.date]
    if args.update_market:
        for d in datelist:
            print(f'更新市场情绪缓存 {d}...')
            md = get_market_sentiment(d)
            if md is not None:
                print('市场情绪缓存已更新')
        print('所有日期数据处理完成！')
    elif args.update_board:
        for d in datelist:
            print(f'更新板块资金与涨幅 {d} 及近3日...')
            dates = trading_recent_dates(d, 3)
            ok = save_board_money_pct(dates)
            print('板块资金与涨幅已更新' if ok else '板块资金与涨幅更新失败')
        print('所有日期数据处理完成！')
    elif args.update_board_params:
        from .board_params_fetch import fetch_and_write
        for d in datelist:
            print(f'参数查询写入板块资金与涨幅 {d}...')
            ok = fetch_and_write(d)
            print('板块资金与涨幅已更新(参数查询)' if ok else '板块资金与涨幅更新失败(参数查询)')
        print('所有日期数据处理完成！')
    elif args.probe_zhishu:
        for d in datelist:
            print(f"测试 zhishu 三字段并写入 {d}...")
            ok = probe_zhishu_store(d)
            print('板块资金与涨幅已更新(探测)' if ok else '探测失败，未写入')
        print('所有日期数据处理完成！')
    elif args.ingest_probe_zhishu:
        for d in datelist:
            print(f"导入探测结果完善 {d}...")
            ok = ingest_probe_zhishu(d)
            print('板块资金与涨幅已更新(导入)' if ok else '导入失败，未写入')
        print('所有日期数据处理完成！')
    elif args.update_board_from_cache:
        days = args.board_days if args.board_days and args.board_days > 0 else 3
        dates = get_cached_dates(days)
        if args.dates:
            dates = [x.strip() for x in args.dates.split(',') if x.strip()]
        if not dates and args.date:
            dates = trading_recent_dates(args.date, days)
        print(f'按缓存交易日更新板块资金与涨幅: {dates} ...')
        ok = save_board_money_pct(dates)
        print('板块资金与涨幅已更新' if ok else '板块资金与涨幅更新失败')
        print('所有日期数据处理完成！')
    elif args.update_market_from_probe:
        for d in datelist:
            print(f'从探测文件更新市场情绪 {d}...')
            ok = update_market_from_probe(d)
            print('市场情绪缓存已更新(探测导入)' if ok else '更新失败')
        print('所有日期数据处理完成！')
    elif args.ingest_market_from_fetch:
        for d in datelist:
            print(f'从fetch文件更新市场情绪 {d}...')
            ok = ingest_market_from_fetch(d)
            print('市场情绪缓存已更新(fetch导入)' if ok else '更新失败')
        print('所有日期数据处理完成！')
    elif args.update_market_counts:
        for d in datelist:
            print(f'更新同花顺全A家数 {d}...')
            ok = update_market_counts(d)
            print('市场情绪家数已更新' if ok else '更新失败')
        print('所有日期数据处理完成！')
    elif args.cache_only:
        for d in datelist:
            print(f'使用缓存生成报告 {d}...')
            market_data = load_market_cache(d)
            try:
                df = pd.read_csv(out_path(f'增强涨停{d}.csv'), encoding='utf-8-sig')
            except Exception:
                df = None
            try:
                df_down = pd.read_csv(out_path(f'增强跌停{d}.csv'), encoding='utf-8-sig')
            except Exception:
                df_down = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                report = generate_daily_analysis_report(df, d, market_data if isinstance(market_data, pd.DataFrame) else None, df_down if isinstance(df_down, pd.DataFrame) else None)
                print(report)
                with open(out_path(f'每日分析报告{d}.md'), 'w', encoding='utf-8') as f:
                    f.write(report)
            else:
                print(f'增强数据缺失，无法生成 {d} 报告')
        print('所有日期数据处理完成！')
    else:
        for d in datelist:
            print(f'正在处理 {d} 数据...')
            process_date(d)
            time.sleep(1)
        print('所有日期数据处理完成！')