import os
import json
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.dirname(__file__)

def load_cfg():
    p = os.path.join(BASE_DIR, 'concept_config.json')
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

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

def score_row(r, weights):
    tech = 0.0
    if not pd.isna(r.get('量比')):
        tech += min(float(r['量比'])/2*15, 15)
    if not pd.isna(r.get('换手率')):
        tech += min(float(r['换手率'])/15*10, 10)
    if not pd.isna(r.get('振幅')):
        tech += min(float(r['振幅'])/12*5, 5)
    money = 0.0
    if not pd.isna(r.get('最新dde大单净额')):
        money += min(max(float(r['最新dde大单净额'])/5e7, 0), 15)
    if not pd.isna(r.get('涨停封单量占成交量比')):
        money += min(float(r['涨停封单量占成交量比'])/15*10, 10)
    if not pd.isna(r.get('涨停封单额')):
        money += min(float(r['涨停封单额'])/1e8*10, 10)
    concept = 0.0
    if not pd.isna(r.get('热门概念数量')):
        concept += min(float(r['热门概念数量'])*5, 15)
    if not pd.isna(r.get('所属概念数量')):
        concept += min(float(r['所属概念数量'])/25*10, 10)
    boardq = 0.0
    if not pd.isna(r.get('涨停开板次数')):
        boardq += max(10 - float(r['涨停开板次数'])*2, 0)
    if not pd.isna(r.get('连续涨停天数')):
        boardq += min(float(r['连续涨停天数'])*2, 5)
    return tech*weights['tech'] + money*weights['money'] + concept*weights['concept'] + boardq*weights['board']

def load_enhanced(date):
    p = os.path.join(BASE_DIR, f'增强涨停{date}.csv')
    try:
        df = pd.read_csv(p, encoding='utf-8-sig')
        return df
    except Exception:
        return None

def next_day_perf(df_today, df_next):
    if df_today is None or df_next is None:
        return pd.DataFrame()
    cols = df_today.columns
    name_col = '股票简称' if '股票简称' in cols else None
    pct_col_next = '最新涨跌幅' if ('最新涨跌幅' in df_next.columns) else None
    if name_col is None or pct_col_next is None:
        return pd.DataFrame()
    df_today = df_today.copy()
    df_next = df_next.copy()
    df_next = df_next[[name_col, pct_col_next]].rename(columns={pct_col_next: '次日涨跌幅'})
    merged = pd.merge(df_today, df_next, on=name_col, how='left')
    return merged

def run_backtest(dates, weight_sets):
    rows = []
    for i in range(len(dates)-1):
        d = dates[i]
        dn = dates[i+1]
        df_d = load_enhanced(d)
        df_dn = load_enhanced(dn)
        if df_d is None or df_dn is None or df_d.empty or df_dn.empty:
            continue
        df_d['连板数'] = df_d['几天几板'].apply(parse_boards) if '几天几板' in df_d.columns else 0
        for ws in weight_sets:
            scores = df_d.apply(lambda r: score_row(r, ws), axis=1)
            df_scored = df_d.copy()
            df_scored['回测评分'] = scores
            # 主攻：连板优先、评分降序，取前20
            pick = df_scored.sort_values(['连板数','回测评分'], ascending=[False, False]).head(20)
            perf = next_day_perf(pick[[c for c in ['股票简称','几天几板','回测评分'] if c in pick.columns]], df_dn)
            if perf.empty:
                continue
            perf['次日涨跌幅'] = pd.to_numeric(perf['次日涨跌幅'], errors='coerce')
            ret = perf['次日涨跌幅'].mean()
            pos_rate = (perf['次日涨跌幅'] > 0).mean()
            max_dd = perf['次日涨跌幅'].min()
            rows.append({'日期': d, '权重': ws, '均值(%)': ret, '上涨占比': pos_rate, '最差(%)': max_dd})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates')
    ap.add_argument('--weights')
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.dates:
        dates = [x.strip() for x in args.dates.split(',') if x.strip()]
    else:
        dates = []
    cfg = load_cfg()
    base_w = cfg.get('score_weights', {'tech':0.30,'money':0.35,'concept':0.25,'board':0.10})
    weight_sets = []
    if args.weights:
        try:
            weight_sets = json.loads(args.weights)
        except Exception:
            weight_sets = []
    if not weight_sets:
        weight_sets = [
            base_w,
            {'tech':0.25,'money':0.45,'concept':0.20,'board':0.10},
            {'tech':0.40,'money':0.30,'concept':0.20,'board':0.10}
        ]
    out = run_backtest(dates, weight_sets)
    if not out.empty:
        if args.out:
            try:
                ddir = os.path.dirname(args.out)
                if ddir:
                    os.makedirs(ddir, exist_ok=True)
            except Exception:
                pass
            try:
                out.to_csv(args.out, index=False, encoding='utf-8-sig')
                print(f'Saved backtest results to {args.out}')
            except Exception as e:
                print(f'Failed to save to {args.out}: {e}')
        else:
            print(out.to_string(index=False))
    else:
        print('Backtest result empty')