import pywencai
import pandas as pd
import re

# 测试使用带zhishu参数的查询获取市场情绪数据
date = "20251117"
query = f"{date}日市场情绪,成交额,上涨家数,下跌家数,平盘家数"

print(f"查询语句: {query}")
print("=" * 80)

try:
    # 使用pywencai查询,添加zhishu参数
    result = pywencai.get(query=query, query_type='zhishu', loop=True)
    
    if result is not None:
        if isinstance(result, pd.DataFrame):
            print(f"返回DataFrame,形状: {result.shape}")
            print(f"\n列名:")
            for i, col in enumerate(result.columns):
                print(f"  {i}: {col}")
            
            print(f"\n数据内容:")
            print(result.to_string())
            
            # 清理列名
            pattern = r"\[\d{8}\]"
            result.columns = [re.sub(pattern, "", str(c)).strip() for c in result.columns]
            
            print(f"\n清理后的列名:")
            for i, col in enumerate(result.columns):
                print(f"  {i}: {col}")
            
            print(f"\n清理后的数据:")
            print(result.to_string())
            
        elif isinstance(result, list):
            print(f"返回列表,长度: {len(result)}")
            for i, item in enumerate(result):
                print(f"\n列表项 {i}:")
                if isinstance(item, pd.DataFrame):
                    print(f"  DataFrame形状: {item.shape}")
                    print(f"  列名: {item.columns.tolist()}")
                    print(item.to_string())
                else:
                    print(f"  类型: {type(item)}")
                    print(item)
        else:
            print(f"返回类型: {type(result)}")
            print(result)
    else:
        print("查询返回None")
        
except Exception as e:
    print(f"查询失败: {e}")
    import traceback
    traceback.print_exc()
