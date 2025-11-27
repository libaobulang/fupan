import pandas as pd
import numpy as np
import re
import pywencai

def clean_columns(df):
    pattern = r"\[\d{8}\]"
    df.columns = [re.sub(pattern, "", str(c)).strip() for c in df.columns]
    return df

def fetch_sector_strength_data(date):
    """
    获取板块强弱分析所需数据
    查询语句: "同花顺指数当日成交额,较上一交易日成交额,近三交易日成交额,涨停家数"
    """
    query = f"{date}同花顺指数成交额,{date}前1日同花顺指数成交额"
    try:
        r = pywencai.get(query=query, query_type='zhishu', loop=True)
        
        df = None
        if isinstance(r, pd.DataFrame):
            df = r
        elif isinstance(r, list) and len(r) > 0 and isinstance(r[0], pd.DataFrame):
            df = r[0]
        elif isinstance(r, dict) and 'tableV1' in r and isinstance(r['tableV1'], pd.DataFrame):
            df = r['tableV1']
            
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = clean_columns(df)
            print(f"DEBUG: Columns: {df.columns.tolist()}")
            print(f"DEBUG: First Row Values: {df.iloc[0].tolist()}")
            return df
        return None
    except Exception as e:
        print(f"获取板块强弱数据失败: {e}")
        return None

def analyze_sector_strength(df):
    """
    分析板块强弱
    """
    if df is None or df.empty:
        return None
        
    df = df.copy()
    
    # 1. 识别列名
    cols = df.columns
    name_col = next((c for c in cols if '指数名称' in c or '指数简称' in c), None)
    code_col = next((c for c in cols if '指数代码' in c), None)
    amt_col = next((c for c in cols if '成交额' in c and '较上一交易日' not in c and '近三' not in c), None)
    prev_amt_col = next((c for c in cols if '较上一交易日成交额' in c), None)
    three_day_amt_col = next((c for c in cols if '近三交易日成交额' in c), None)
    zt_col = next((c for c in cols if '涨停家数' in c), None)
    
    if not all([name_col, amt_col, prev_amt_col, zt_col]):
        print("关键列缺失，无法分析")
        return None
        
    # 2. 数据清洗与转换
    def to_num(v):
        return pd.to_numeric(str(v).replace(',','').replace('亿','').replace('万',''), errors='coerce')

    df['成交额'] = df[amt_col].apply(to_num)
    df['昨成交额'] = df[prev_amt_col].apply(to_num)
    df['涨停家数'] = df[zt_col].apply(to_num).fillna(0)
    
    # 3. 计算指标
    # 成交额增幅
    df['成交额增幅'] = (df['成交额'] - df['昨成交额']) / (df['昨成交额'] + 1) 
    
    # 4. 计算强弱得分
    # 逻辑: 
    # - 涨停家数权重高 (代表情绪极致)
    # - 成交额增幅权重中 (代表资金关注度提升)
    # - 绝对成交额权重低 (代表板块容量)
    
    # 归一化
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-5)
    
    df['涨停得分'] = normalize(df['涨停家数']) * 100
    df['增幅得分'] = normalize(df['成交额增幅']) * 100
    df['成交得分'] = normalize(df['成交额']) * 100
    
    # 综合得分 = 涨停*0.5 + 增幅*0.3 + 成交*0.2
    df['强弱得分'] = df['涨停得分'] * 0.5 + df['增幅得分'] * 0.3 + df['成交得分'] * 0.2
    
    # 5. 排序与分类
    df = df.sort_values('强弱得分', ascending=False)
    
    # 选取前20名
    top_df = df.head(20).copy()
    
    result_cols = [name_col, code_col, '强弱得分', '成交额', '成交额增幅', '涨停家数']
    return top_df[result_cols]

def format_report(df):
    if df is None or df.empty:
        return "无板块强弱数据"
        
    lines = []
    lines.append("### 板块强弱分析 (基于资金与涨停)")
    lines.append("| 板块名称 | 强弱得分 | 成交额(亿) | 增幅(%) | 涨停家数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    
    for _, row in df.iterrows():
        name = row.iloc[0]
        score = f"{row['强弱得分']:.1f}"
        amt = f"{row['成交额']/1e8:.2f}" if row['成交额'] > 1e8 else f"{row['成交额']/1e4:.0f}万"
        growth = f"{row['成交额增幅']*100:.1f}"
        zt = int(row['涨停家数'])
        
        lines.append(f"| {name} | {score} | {amt} | {growth} | {zt} |")
        
    return "\n".join(lines)

if __name__ == "__main__":
    # 测试代码
    date = "20241127" # 使用一个假设的日期或最近交易日
    print(f"Fetching data for {date}...")
    df = fetch_sector_strength_data(date)
    if df is not None:
        print(f"Fetched {len(df)} rows.")
        res = analyze_sector_strength(df)
        print(format_report(res))
    else:
        print("Fetch failed.")
