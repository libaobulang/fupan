import akshare as ak
import traceback

# 测试获取上证指数数据
print("测试获取上证指数历史数据...")
try:
    df = ak.index_zh_a_hist(symbol="000001", period="daily", start_date="20251122", end_date="20251122")
    print(f"成功! 返回数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    if not df.empty:
        print(f"\n数据示例:")
        print(df.head())
except Exception as e:
    print(f"失败: {e}")
    traceback.print_exc()
