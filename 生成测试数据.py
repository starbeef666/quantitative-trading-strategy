#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试数据 - 创建符合策略条件的股票数据
"""

import csv
import random
from datetime import datetime, timedelta

def generate_stock_data(filename, market_code, num_stocks=50, days_per_stock=200):
    """生成股票数据"""
    
    # 生成日期序列
    start_date = datetime(2020, 1, 1)
    dates = []
    for i in range(days_per_stock):
        date = start_date + timedelta(days=i)
        if date.weekday() < 5:  # 只包含工作日
            dates.append(date.strftime('%Y%m%d'))
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'])
        
        for stock_idx in range(num_stocks):
            # 生成股票代码
            if market_code == 'SH':
                ts_code = f"60{stock_idx:04d}.SH"
            else:
                ts_code = f"00{stock_idx:04d}.SZ"
            
            # 生成基础价格
            base_price = random.uniform(10, 50)
            current_price = base_price
            
            for day_idx, date in enumerate(dates):
                # 生成价格波动
                if day_idx == 0:
                    pre_close = base_price
                else:
                    pre_close = current_price
                
                # 生成价格变化
                change_pct = random.uniform(-0.05, 0.05)  # -5% 到 +5%
                change = pre_close * change_pct
                current_price = pre_close + change
                
                # 生成OHLC
                open_price = pre_close * random.uniform(0.98, 1.02)
                close_price = current_price
                high_price = max(open_price, close_price) * random.uniform(1.0, 1.03)
                low_price = min(open_price, close_price) * random.uniform(0.97, 1.0)
                
                # 生成成交量
                vol = random.randint(100000, 2000000)
                amount = vol * close_price
                
                # 计算涨跌幅
                pct_chg = (close_price - pre_close) / pre_close * 100
                
                writer.writerow([
                    ts_code, date, round(open_price, 2), round(high_price, 2),
                    round(low_price, 2), round(close_price, 2), round(pre_close, 2),
                    round(change, 2), round(pct_chg, 2), vol, int(amount)
                ])

def generate_strategy_signals(filename, market_code, num_signals=1000):
    """生成符合策略条件的信号数据"""
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'])
        
        for signal_idx in range(num_signals):
            # 生成股票代码
            if market_code == 'SH':
                ts_code = f"60{signal_idx:04d}.SH"
            else:
                ts_code = f"00{signal_idx:04d}.SZ"
            
            # 生成V型底和三连涨模式
            base_price = random.uniform(15, 30)
            
            # 第0天：最低点（V型底）
            close_0 = base_price
            
            # 第1天：开始上涨
            close_1 = close_0 * random.uniform(1.02, 1.05)
            
            # 第2天：继续上涨
            close_2 = close_1 * random.uniform(1.02, 1.05)
            
            # 第3天：继续上涨（三连涨）
            close_3 = close_2 * random.uniform(1.02, 1.05)
            
            # 生成交易日期
            start_date = datetime(2020, 1, 1)
            trade_date = (start_date + timedelta(days=signal_idx)).strftime('%Y%m%d')
            
            # 生成OHLC
            open_price = close_3 * random.uniform(0.98, 1.02)
            high_price = max(open_price, close_3) * random.uniform(1.0, 1.03)
            low_price = min(open_price, close_3) * random.uniform(0.97, 1.0)
            
            # 生成成交量
            vol = random.randint(500000, 3000000)
            amount = vol * close_3
            
            # 计算涨跌幅
            pct_chg = (close_3 - close_2) / close_2 * 100
            
            writer.writerow([
                ts_code, trade_date, round(open_price, 2), round(high_price, 2),
                round(low_price, 2), round(close_3, 2), round(close_2, 2),
                round(close_3 - close_2, 2), round(pct_chg, 2), vol, int(amount)
            ])

def main():
    """主函数"""
    print("开始生成测试数据...")
    
    # 生成上海市场数据
    print("生成上海市场数据...")
    generate_stock_data('上海测试数据.csv', 'SH', num_stocks=100, days_per_stock=300)
    
    # 生成深圳市场数据
    print("生成深圳市场数据...")
    generate_stock_data('深圳测试数据.csv', 'SZ', num_stocks=100, days_per_stock=300)
    
    # 生成符合策略条件的信号数据
    print("生成策略信号数据...")
    generate_strategy_signals('上海策略信号.csv', 'SH', num_signals=5000)
    generate_strategy_signals('深圳策略信号.csv', 'SZ', num_signals=5000)
    
    print("数据生成完成！")
    print("文件列表：")
    print("- 上海测试数据.csv")
    print("- 深圳测试数据.csv")
    print("- 上海策略信号.csv")
    print("- 深圳策略信号.csv")

if __name__ == "__main__":
    main()