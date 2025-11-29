import pywencai
import pandas as pd
from datetime import datetime

def test_fetch(date):
    print(f"Testing fetch for {date}...")
    # Original query from fetch_sector_strength_data
    query = f"{date}同花顺指数当日成交额,{date}同花顺指数较上一交易日成交额,{date}同花顺指数近三交易日成交额,{date}同花顺指数涨停家数"
    # Adding 资金流向
    query_with_flow = f"{query},{date}同花顺指数资金流向"
    
    try:
        print(f"Querying: {query_with_flow}")
        r = pywencai.get(query=query_with_flow, query_type='zhishu', loop=True)
        if isinstance(r, pd.DataFrame) and not r.empty:
            print("Fetch successful!")
            print("Columns:", r.columns.tolist())
            print(r.head())
        else:
            print("Fetch returned no DataFrame or empty.")
            print("Result type:", type(r))
    except Exception as e:
        print(f"Fetch failed: {e}")

if __name__ == "__main__":
    test_fetch(20251128)
