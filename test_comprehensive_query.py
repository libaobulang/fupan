import pywencai
import pandas as pd
import re
import sys
import os

# 再次简化版测试脚本
date = "20251117"
query = f"{date}日上证指数涨跌幅,{date}日深证成指涨跌幅,{date}日创业板指涨跌幅,{date}日同花顺全A(沪深京)的涨停家数,{date}日跌停家数,{date}日上涨家数,{date}日下跌家数,{date}日平盘家数,{date}日A股总成交额"

print("START_TEST")
try:
    # 尝试屏蔽标准输出以减少干扰(可能无效如果pywencai用子进程)
    result = pywencai.get(query=query, query_type='zhishu', loop=True)
    
    print(f"RESULT_TYPE: {type(result)}")
    
    if isinstance(result, pd.DataFrame):
        print(f"DATAFRAME_SHAPE: {result.shape}")
        print("COLUMNS:")
        for c in result.columns:
            print(f"  {c}")
    elif isinstance(result, dict):
        print(f"DICT_KEYS: {list(result.keys())}")
        for k, v in result.items():
            print(f"KEY: {k}, TYPE: {type(v)}")
            if isinstance(v, pd.DataFrame):
                print(f"  DF_SHAPE: {v.shape}")
                print("  DF_COLUMNS:")
                for c in v.columns:
                    print(f"    {c}")

except Exception as e:
    print(f"ERROR: {e}")
print("END_TEST")
