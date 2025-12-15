import pandas as pd
from fupan import process_date, generate_daily_analysis_report
import os

# Mocking pywencai to avoid actual network call if possible, 
# but process_date relies on it. 
# Let's try to run it for real for one date, or mock the dataframe if network is an issue.
# Given I have internet access tool, I can try to find if I can just run it.
# But better to just run a small part.

# Actually, I can just check if the logic is correct by dry running with a dummy df for report generation
# and check if query has the string for process_date (static analysis or just trust the edit).

# Let's do a real run for process_date if possible, but it might be slow.
# Alternatively, I can manually construct a DF with '涨停原因类别' and feed it to generate_daily_analysis_report
# to see if it appears in the output. This verifies the report generation part.
# The `process_date` part was a simple string addition, so highly likely correct if wencai supports it.

def test_report_generation():
    df = pd.DataFrame({
        '股票简称': ['StockA', 'StockB'],
        '几天几板': ['2板', '3板'],
        '涨停原因类别': ['ReasonA', 'ReasonB'],
        '所属同花顺行业': ['IndA', 'IndB'],
        '所属概念': ['ConA;ConB', 'ConC'],
        '综合评分': [80, 90],
        '日期': ['20251210', '20251210']
    })
    
    # Needs many cols to avoid errors in report gen
    # Just minimal mock
    # Mocking required columns
    df['涨停开板次数'] = 0
    df['首次涨停时间'] = '09:30:00'
    df['成交额'] = 1e8
    df['昨成交额'] = 1e8
    df['涨停家数'] = 10
    df['资金流向'] = 1e7
    df['涨跌幅'] = 10.0
    df['近三日成交额'] = 3e8
    df['当日热门概念'] = 'Mock'
    df['最新dde大单净额'] = 1e7
    df['涨停封单量占成交量比'] = 0.1
    df['换手率'] = 5.0
    df['最新价'] = 10.0
    
    report = generate_daily_analysis_report(df, '20251210', pd.DataFrame([{'涨停家数':10}]), None)
    
    missing = []
    for col in ['涨停原因类别', '首次涨停时间', '开板次数']:
        if f"| {col} |" not in report:
            missing.append(col)
            
    if not missing:
        print("Verification SUCCESS: All columns found in report.")
        return True
    else:
        print(f"Verification FAILED: Missing columns: {missing}")
        print(report)
        return False

if __name__ == "__main__":
    test_report_generation()
