# Sector Strength Analysis Functions
# These functions fetch and analyze sector strength data using zhishu query

def fetch_sector_strength_data(date):
    """
    获取板块强弱分析所需数据
    查询语句: "同花顺指数当日成交额,较上一交易日成交额,近三交易日成交额,涨停家数"
    """
    query = f"{date}同花顺指数当日成交额,{date}同花顺指数较上一交易日成交额,{date}同花顺指数近三交易日成交额,{date}同花顺指数涨停家数"
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
            return df
        return None
    except Exception:
        return None

def analyze_sector_strength(df, date):
    """
    分析板块强弱
    """
    if df is None or df.empty:
        return None
    df = df.copy()
    raw_cols = df.columns.tolist()
    # 识别列名
    name_col = next((c for c in raw_cols if '指数名称' in c or '指数简称' in c), None)
    code_col = next((c for c in raw_cols if '指数代码' in c), None)
    # 解析日期
    try:
        dt = datetime.strptime(str(date), '%Y%m%d')
        date_str = dt.strftime('%Y%m%d')
    except:
        date_str = str(date)
    # 查找成交额列
    amt_col = next((c for c in raw_cols if '成交额' in c and date_str in c and '区间' not in c and '较上一' not in c), None)
    if not amt_col:
        amt_cols = [c for c in raw_cols if '成交额' in c and '区间' not in c and '较上一' not in c]
        if len(amt_cols) >= 2:
            amt_cols.sort()
            amt_col = amt_cols[-1]
        elif len(amt_cols) == 1:
            amt_col = amt_cols[0]
    prev_amt_col = next((c for c in raw_cols if '较上一交易日成交额' in c), None)
    if not prev_amt_col:
        candidates = [c for c in raw_cols if '成交额' in c and '区间' not in c and c != amt_col]
        if candidates:
            prev_amt_col = candidates[-1]
    three_day_amt_col = next((c for c in raw_cols if '近三交易日' in c or '区间成交额' in c), None)
    zt_col = next((c for c in raw_cols if '涨停家数' in c), None)
    if not all([name_col, amt_col, prev_amt_col, three_day_amt_col, zt_col]):
        return None
    # 数据清洗与转换
    def to_num(v):
        try:
            s = str(v).replace(',','').replace('亿','').replace('万','')
            return float(s)
        except:
            return 0.0
    df['成交额'] = df[amt_col].apply(to_num)
    df['昨成交额'] = df[prev_amt_col].apply(to_num)
    df['近三日成交额'] = df[three_day_amt_col].apply(to_num)
    df['涨停家数'] = df[zt_col].apply(to_num).fillna(0)
    # 计算指标
    df['成交额增幅'] = (df['成交额'] - df['昨成交额']) / (df['昨成交额'] + 1)
    df['趋势强度'] = df['成交额'] / ((df['近三日成交额'] / 3) + 1)
    # 计算强弱得分
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-5)
    df['涨停得分'] = normalize(df['涨停家数']) * 100
    df['增幅得分'] = normalize(df['成交额增幅']) * 100
    df['趋势得分'] = normalize(df['趋势强度']) * 100
    df['强弱得分'] = df['涨停得分'] * 0.4 + df['增幅得分'] * 0.3 + df['趋势得分'] * 0.3
    # 排序与分类
    df = df.sort_values('强弱得分', ascending=False)
    top_df = df.head(20).copy()
    result_cols = [name_col, code_col, '强弱得分', '成交额', '成交额增幅', '涨停家数', '趋势强度']
    return top_df[result_cols]

def format_sector_strength_report(df):
    """格式化板块强弱分析报告"""
    if df is None or df.empty:
        return "无板块强弱数据"
    lines = []
    lines.append("### 板块强弱分析 (基于资金与涨停)")
    lines.append("| 板块名称 | 强弱得分 | 成交额(亿) | 增幅(%) | 涨停家数 | 趋势强度 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in df.iterrows():
        name = row.iloc[0]
        score = f"{row['强弱得分']:.1f}"
        amt = f"{row['成交额']/1e8:.2f}" if row['成交额'] > 1e8 else f"{row['成交额']/1e4:.0f}万"
        growth = f"{row['成交额增幅']*100:.1f}"
        zt = int(row['涨停家数'])
        trend = f"{row['趋势强度']:.2f}"
        lines.append(f"| {name} | {score} | {amt} | {growth} | {zt} | {trend} |")
    return "\n".join(lines)
