
import sys
import os
sys.path.append(os.path.dirname(__file__))

from fetch_market import fetch_market_zhishu

date = '20251128'
print(f"Testing fetch_market_zhishu for {date}...")
result = fetch_market_zhishu(date)
print("Result:")
for k, v in result.items():
    print(f"{k}: {v}")
