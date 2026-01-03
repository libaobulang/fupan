
import csv
import os

# Configuration
report_path = r'd:\fupan\reports\每日分析报告20260103.md'
csv_path = r'd:\fupan\data\增强涨停20260103.csv'

# Target structure
# Current: | 股票简称 | 股票代码 | 几天几板 | 所属同花顺行业 | 涨停原因类别 | 首次涨停时间 | 开板次数 | 所属概念数量 | 所属概念 |
# New:     | 股票简称 | 股票代码 | 股票价格 | 几天几板 | 流通股 | 换手率 | 封成比 | 所属同花顺行业 | 涨停原因类别 | 首次涨停时间 | 开板次数 | 所属概念数量 | 所属概念 |

def format_float(val, precision=2):
    try:
        return f"{float(val):.{precision}f}"
    except (ValueError, TypeError):
        return str(val)

def calculate_shares(market_cap_str, price_str):
    try:
        cap = float(market_cap_str)
        price = float(price_str)
        if price == 0: return ""
        shares = cap / price
        return f"{shares/1e8:.2f}亿"
    except (ValueError, TypeError):
        return ""

def main():
    if not os.path.exists(csv_path) or not os.path.exists(report_path):
        print(f"File not found: {csv_path} or {report_path}")
        return

    # 1. Read CSV Data
    data_map = {} # Name -> Data Dict
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('股票简称', '').strip()
            if name:
                if '最新dde大单净额' not in row and 'dde大单净额' in row:
                    row['最新dde大单净额'] = row['dde大单净额']
                data_map[name] = row

    # 2. Read Markdown Report
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3. Process Lines
    new_lines = []
    in_table = False
    header_processed = False
    
    # New Headers
    new_headers = ['股票简称', '股票代码', '股票价格', '几天几板', '流通股', '换手率', '封成比', '所属同花顺行业', '涨停原因类别', '首次涨停时间', '开板次数', '所属概念数量', '所属概念']
    
    separator_line = '| ' + ' | '.join(['---'] * len(new_headers)) + ' |'

    for line in lines:
        stripped = line.strip()
        
        # Detect Table Start (Header Row)
        if stripped.startswith('|') and '股票简称' in stripped and '几天几板' in stripped:
            in_table = True
            header_processed = True
            new_lines.append('| ' + ' | '.join(new_headers) + ' |\n')
            continue

        # Detect Separator Row
        if in_table and set(stripped.replace('|', '').replace('-', '').replace(' ', '')) == set():
             new_lines.append(separator_line + '\n')
             continue

        # Detect Data Rows
        if in_table and stripped.startswith('|'):
            # Parse existing row
            parts = [p.strip() for p in stripped.split('|') if p.strip() != '']
            
            # Extract Key Info from existing row to match CSV
            # Need to be careful about mapping. 
            # Existing parts (based on viewed file):
            # 0: Name, 1: Code, 2: Days, 3: Industry, 4: Reason, 5: Time, 6: OpenCount, 7: ConceptCount, 8: Concepts
            
            if len(parts) >= 9:
                name = parts[0]
                code = parts[1]
                days = parts[2]
                industry = parts[3]
                reason = parts[4]
                time = parts[5]
                open_count = parts[6]
                concept_count = parts[7]
                concepts = parts[8]

                # Lookup new data
                row_data = data_map.get(name)
                
                price = ""
                turnover = ""
                ratio = ""
                shares = ""
                
                if row_data:
                    price = format_float(row_data.get('最新价', ''))
                    turnover = format_float(row_data.get('换手率', ''))
                    ratio_raw = row_data.get('涨停封单量占成交量比', '')
                    ratio = format_float(ratio_raw) if ratio_raw else ""
                    
                    mcap = row_data.get('a股市值(不含限售股)', '')
                    shares = calculate_shares(mcap, price)
                
                # Construct new row
                new_row = [
                    name,
                    code,
                    price,
                    days,
                    shares,
                    turnover,
                    ratio,
                    industry,
                    reason,
                    time,
                    open_count,
                    concept_count,
                    concepts
                ]
                
                new_line = '| ' + ' | '.join(new_row) + ' |\n'
                new_lines.append(new_line)
            else:
                # Malformed row, keep as is or skip? Keep as is but might break table
                new_lines.append(line)
        else:
            if in_table and not stripped.startswith('|'):
                in_table = False # End of table
            new_lines.append(line)

    # 4. Write Back
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("Report updated successfully.")

if __name__ == '__main__':
    main()
