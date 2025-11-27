import pandas as pd
import pywencai
import numpy as np
import re
import os
import argparse
from datetime import datetime
import akshare as ak

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
    """使用 AkShare 获取指定日期的指数涨跌幅和成交额"""
    result = {
        '上证指数涨跌幅': np.nan,
        '深证成指涨跌幅': np.nan,
        '创业板指涨跌幅': np.nan,
        'A股总成交额': np.nan,
    }
    
    sh_amount = 0
    sz_amount = 0
    
    # 获取上证指数数据
    try:
        df_sh = ak.index_zh_a_hist(symbol="000001", period="daily", start_date=date, end_date=date)
        if not df_sh.empty:
            result['上证指数涨跌幅'] = to_num(df_sh['涨跌幅'].iloc[0])
            sh_amount = to_num(df_sh['成交额'].iloc[0]) / 100000000  # 转换为亿元
    except Exception as e:
        print(f'获取上证指数失败: {e}')
    
    # 获取深证成指数据
    try:
        df_sz = ak.index_zh_a_hist(symbol="399001", period="daily", start_date=date, end_date=date)
        if not df_sz.empty:
            result['深证成指涨跌幅'] = to_num(df_sz['涨跌幅'].iloc[0])
            sz_amount = to_num(df_sz['成交额'].iloc[0]) / 100000000  # 转换为亿元
    except Exception as e:
        print(f'获取深证成指失败: {e}')
    
    # 获取创业板指数据
    try:
        df_cyb = ak.index_zh_a_hist(symbol="399006", period="daily", start_date=date, end_date=date)
        if not df_cyb.empty:
            result['创业板指涨跌幅'] = to_num(df_cyb['涨跌幅'].iloc[0])
    except Exception as e:
        print(f'获取创业板指失败: {e}')
    
    # 计算A股总成交额(上证+深证)
    if sh_amount > 0 or sz_amount > 0:
        result['A股总成交额'] = sh_amount + sz_amount
    
    return result

def fetch_market_zhishu(date):
    """使用综合查询获取市场情绪数据(包括指数涨跌幅、成交额、涨跌停家数等)"""
    result = {
        '上证指数涨跌幅': np.nan,
        '深证成指涨跌幅': np.nan,
        '创业板指涨跌幅': np.nan,
        'A股总成交额': np.nan,
        '上涨家数': np.nan,
        '下跌家数': np.nan,
        '平盘家数': np.nan,
        '涨停家数': np.nan,
        '跌停家数': np.nan,
    }
    
    query = f"{date}日上证指数涨跌幅,{date}日深证成指涨跌幅,{date}日创业板指涨跌幅,{date}日同花顺全A(沪深京)的涨停家数,{date}日跌停家数,{date}日上涨家数,{date}日下跌家数,{date}日平盘家数,{date}日A股总成交额"
    
    try:
        r = pywencai.get(query=query, query_type='zhishu', loop=True)
        
        # 处理返回结果可能是字典或列表的情况
        if isinstance(r, dict):
            # 如果是字典，尝试获取第一个DataFrame值
            for v in r.values():
                if isinstance(v, pd.DataFrame):
                    r = v
                    break
        elif isinstance(r, list) and len(r) > 0:
            # 如果是列表，尝试获取第一个DataFrame项
            if isinstance(r[0], pd.DataFrame):
                r = r[0]
        
        if isinstance(r, pd.DataFrame) and not r.empty:
            r = clean_columns(r)
            
            # 调试打印到文件
            print("DEBUG: 开始写入调试日志...")
            try:
                log_path = os.path.join(os.path.dirname(__file__), 'debug_log.txt')
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"DEBUG: Columns: {r.columns.tolist()}\n")
                    
                    # 遍历每一行数据
                    for idx, row in r.iterrows():
                        # 获取指数代码或名称来判断是哪一行
                        code = str(row.get('指数代码', ''))
                        name = str(row.get('指数简称', ''))
                        f.write(f"DEBUG: Row {idx}: Code={code}, Name={name}\n")
                        for c in r.columns:
                            f.write(f"  {c}: {row[c]}\n")
                print(f"DEBUG: 调试日志已写入: {log_path}")
            except Exception as e:
                print(f"写入调试日志失败: {e}")

            # 遍历每一行数据
            for idx, row in r.iterrows():
                # 获取指数代码或名称来判断是哪一行
                code = str(row.get('指数代码', ''))
                name = str(row.get('指数简称', ''))
                
                # 提取同花顺全A的数据 (883957)
                if '883957' in code or '同花顺全A' in name:
                    # 提取成交额
                    for c in r.columns:
                        n = str(c)
                        if '成交额' in n and '指数@成交额' in n:
                            val = to_num(row[c])
                            if pd.notna(val):
                                result['A股总成交额'] = val / 100000000 # 转换为亿元
                        elif '上涨家数' in n and '指数@上涨家数' in n:
                            result['上涨家数'] = to_num(row[c])
                        elif '下跌家数' in n and '指数@下跌家数' in n:
                            result['下跌家数'] = to_num(row[c])
                        elif '平盘家数' in n and '指数@平盘家数' in n:
                            result['平盘家数'] = to_num(row[c])
                        elif '涨停家数' in n and '指数@涨停家数' in n:
                            result['涨停家数'] = to_num(row[c])
                        elif '跌停家数' in n and '指数@跌停家数' in n:
                            result['跌停家数'] = to_num(row[c])
                            
                # 提取上证指数数据 (000001)
                elif '000001' in code or '上证指数' in name:
                    for c in r.columns:
                        if '涨跌幅' in str(c):
                            result['上证指数涨跌幅'] = to_num(row[c])
                            
                # 提取深证成指数据 (399001)
                elif '399001' in code or '深证成指' in name:
                    for c in r.columns:
                        if '涨跌幅' in str(c):
                            result['深证成指涨跌幅'] = to_num(row[c])
                            
                # 提取创业板指数据 (399006)
                elif '399006' in code or '创业板指' in name:
                    for c in r.columns:
                        if '涨跌幅' in str(c):
                            result['创业板指涨跌幅'] = to_num(row[c])

            print(f"zhishu查询成功:")
            print(f"  指数涨跌幅: 上证={result['上证指数涨跌幅']}%, 深证={result['深证成指涨跌幅']}%, 创业板={result['创业板指涨跌幅']}%")
            print(f"  市场情绪: 成交额={result['A股总成交额']:.2f}亿, 上涨={result['上涨家数']}, 下跌={result['下跌家数']}, 涨停={result['涨停家数']}, 跌停={result['跌停家数']}")
            
    except Exception as e:
        print(f'zhishu查询失败: {e}')
        import traceback
        traceback.print_exc()
    
    return result

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
        # 使用综合查询获取所有数据
        market_data = fetch_market_zhishu(d)
        
        # 如果综合查询失败(关键数据缺失),尝试回退到旧方法(虽然旧方法已被移除,但为了兼容性保留接口调用结构,实际这里主要依赖综合查询)
        # 注意: fetch_indices 和 fetch_breadth 函数仍保留在文件中,如果需要可以作为备选,但目前逻辑主要依赖 fetch_market_zhishu
        
        out = merge_cache(d, market_data)
        print(out.to_string(index=False))

if __name__ == '__main__':
    main()