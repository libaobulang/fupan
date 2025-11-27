import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# 添加当前目录到path以导入fetch_market
sys.path.append(os.path.dirname(__file__))
from fetch_market import fetch_market_zhishu, fetch_indices, fetch_breadth

TARGET_FILE = r'd:\fupan\data\cache_market\市场情绪.csv'

def calculate_sentiment(row):
    try:
        up = pd.to_numeric(row['上涨家数'], errors='coerce')
        down = pd.to_numeric(row['下跌家数'], errors='coerce')
        zt = pd.to_numeric(row['涨停家数'], errors='coerce')
        dt = pd.to_numeric(row['跌停家数'], errors='coerce')
        
        if pd.isna(up) or pd.isna(down) or pd.isna(zt) or pd.isna(dt):
            return np.nan
            
        total = up + down
        if total == 0:
            return np.nan
            
        up_ratio = up / total
        
        # 情绪分数计算公式
        # 涨跌比权重 0.4
        # 涨停家数权重 0.3 (100家涨停满分)
        # 跌停家数权重 0.3 (0家跌停满分, 50家跌停0分)
        
        s_up = up_ratio * 0.4
        s_zt = min(max(zt / 100, 0), 1) * 0.3
        s_dt = (1 - min(max(dt / 50, 0), 1)) * 0.3
        
        return (s_up + s_zt + s_dt) * 100
    except Exception:
        return np.nan

def update_history():
    # 读取现有文件
    if os.path.exists(TARGET_FILE):
        df = pd.read_csv(TARGET_FILE)
    else:
        print(f"文件不存在: {TARGET_FILE}")
        return

    # 确保日期列是字符串类型
    df['日期'] = df['日期'].astype(str)
    
    # 需要更新的日期范围
    dates = [d.strftime('%Y%m%d') for d in pd.date_range(start='20251117', end='20251126')]
    
    for date in dates:
        print(f"正在处理日期: {date}...")
        
        # 1. 获取市场情绪数据 (成交额, 上涨, 下跌, 平盘)
        market_data = fetch_market_zhishu(date)
        
        # 2. 获取指数数据
        indices_data = fetch_indices(date)
        
        # 3. 获取涨跌停数据 (如果zhishu查询没返回涨跌家数, 这里也会尝试获取)
        breadth_data = fetch_breadth(date)
        
        # 合并数据
        # 优先使用 market_data 的上涨/下跌/平盘
        # 优先使用 breadth_data 的涨停/跌停
        
        # 查找或创建行
        mask = df['日期'] == date
        if not mask.any():
            new_row = pd.DataFrame({'日期': [date]})
            df = pd.concat([df, new_row], ignore_index=True)
            mask = df['日期'] == date
            
        idx = df.index[mask][0]
        
        # 更新指数数据
        if pd.notna(indices_data.get('上证指数涨跌幅')):
            df.at[idx, '上证指数涨跌幅'] = indices_data['上证指数涨跌幅']
        if pd.notna(indices_data.get('深证成指涨跌幅')):
            df.at[idx, '深证成指涨跌幅'] = indices_data['深证成指涨跌幅']
        if pd.notna(indices_data.get('创业板指涨跌幅')):
            df.at[idx, '创业板指涨跌幅'] = indices_data['创业板指涨跌幅']
            
        # 更新成交额
        if pd.notna(market_data.get('A股总成交额')):
            df.at[idx, 'A股总成交额(元)'] = market_data['A股总成交额'] * 100000000 # 还原为元
        elif pd.notna(indices_data.get('A股总成交额')):
             df.at[idx, 'A股总成交额(元)'] = indices_data['A股总成交额'] * 100000000
             
        # 更新家数
        if pd.notna(market_data.get('上涨家数')):
            df.at[idx, '上涨家数'] = market_data['上涨家数']
        elif pd.notna(breadth_data.get('上涨家数')):
            df.at[idx, '上涨家数'] = breadth_data['上涨家数']
            
        if pd.notna(market_data.get('下跌家数')):
            df.at[idx, '下跌家数'] = market_data['下跌家数']
        elif pd.notna(breadth_data.get('下跌家数')):
            df.at[idx, '下跌家数'] = breadth_data['下跌家数']
            
        if pd.notna(market_data.get('平盘家数')):
            df.at[idx, '平盘家数'] = market_data['平盘家数']
            
        if pd.notna(market_data.get('涨停家数')):
            df.at[idx, '涨停家数'] = market_data['涨停家数']
        elif pd.notna(breadth_data.get('涨停家数')):
            df.at[idx, '涨停家数'] = breadth_data['涨停家数']
            
        if pd.notna(market_data.get('跌停家数')):
            df.at[idx, '跌停家数'] = market_data['跌停家数']
        elif pd.notna(breadth_data.get('跌停家数')):
            df.at[idx, '跌停家数'] = breadth_data['跌停家数']
            
        # 计算情绪分数
        sentiment = calculate_sentiment(df.iloc[idx])
        if pd.notna(sentiment):
            df.at[idx, '情绪分数'] = sentiment
            
        print(f"已更新 {date}: 情绪分数={sentiment}")

    # 保存文件
    df.sort_values('日期', inplace=True)
    df.to_csv(TARGET_FILE, index=False, encoding='utf-8-sig')
    print(f"更新完成,已保存至 {TARGET_FILE}")
    print(df.to_string())

if __name__ == '__main__':
    update_history()
