import pandas as pd
import pywencai
import numpy as np
from datetime import datetime
import re

# 导入板块强弱分析函数
from sector_strength import fetch_sector_strength_data, analyze_sector_strength, format_sector_strength_report

def main():
    date = datetime.now().strftime('%Y%m%d')
    print(f"正在生成{date}的板块强弱分析报告...")
    
    # 获取板块强弱数据
    sector_df = fetch_sector_strength_data(date)
    if sector_df is not None:
        # 分析数据
        sector_analysis = analyze_sector_strength(sector_df, date)
        
        # 生成报告
        if sector_analysis is not None:
            report = format_sector_strength_report(sector_analysis)
            
            # 输出到文件
            output_path = f"data/板块资金及强弱分析{date}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# 板块资金及强弱分析报告 {date}\\n\\n")
                f.write(report)
            
            print(f"报告已生成: {output_path}")
            print("\\n" + "="*50)
            print(report)
            print("="*50)
        else:
            print("板块数据分析失败")
    else:
        print("板块数据获取失败")

if __name__ == "__main__":
    main()
