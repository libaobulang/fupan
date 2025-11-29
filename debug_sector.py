from fupan import fetch_sector_strength_data
import pandas as pd

df = fetch_sector_strength_data(20251128)
if df is not None:
    raw_cols = df.columns.tolist()
    print("All columns:")
    for i, col in enumerate(raw_cols):
        print(f"  {i}: {col}")
    
    print("\nChecking column detection:")
    name_col = next((c for c in raw_cols if '指数名称' in c or '指数简称' in c), None)
    code_col = next((c for c in raw_cols if '指数代码' in c), None)
    amt_col = next((c for c in raw_cols if '成交额' in c and '区间' not in c and '较上一' not in c and '近三' not in c), None)
    prev_amt_col = next((c for c in raw_cols if '较上一交易日成交额' in c or ('较上一' in c and '成交额' in c)), None)
    three_day_amt_col = next((c for c in raw_cols if '近三交易日成交额' in c or ('近三' in c and '成交额' in c) or '区间成交额' in c), None)
    zt_col = next((c for c in raw_cols if '涨停家数' in c), None)
    flow_col = next((c for c in raw_cols if '资金流向' in c), None)
    pct_col = next((c for c in raw_cols if '涨跌幅' in c), None)
    
    print(f"name_col: {name_col}")
    print(f"code_col: {code_col}")
    print(f"amt_col: {amt_col}")
    print(f"prev_amt_col: {prev_amt_col}")
    print(f"three_day_amt_col: {three_day_amt_col}")
    print(f"zt_col: {zt_col}")
    print(f"flow_col: {flow_col}")
    print(f"pct_col: {pct_col}")
    
    print(f"\nAll required? {all([name_col, amt_col, prev_amt_col, three_day_amt_col, zt_col])}")
else:
    print("No data fetched")
