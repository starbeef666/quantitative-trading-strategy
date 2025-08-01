#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化策略测试 - 计算深圳和上海收益率
"""

import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleStrategyTest:
    def __init__(self):
        """初始化策略"""
        self.entry_price = None
        self.remaining_shares = 1.0
        self.total_pnl = 0.0
        self.prev_close = None
        self.is_first_drop = True
        self.stop_loss_price = None
        self.trades = []
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"正在加载数据: {file_path}")
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            print(f"数据加载完成，共 {len(data):,} 行")
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []
        return data
    
    def extract_features(self, stock_data, window=20):
        """提取特征值"""
        features = []
        
        # 按股票代码分组
        stocks = {}
        for row in stock_data:
            ts_code = row.get('ts_code', '')
            if ts_code not in stocks:
                stocks[ts_code] = []
            stocks[ts_code].append(row)
        
        for ts_code, stock in stocks.items():
            # 按日期排序
            stock.sort(key=lambda x: x.get('trade_date', ''))
            
            if len(stock) < window + 5:
                continue
                
            for i in range(window + 3, len(stock) - 2):
                try:
                    # 基础特征
                    close_0 = float(stock[i-3].get('close', 0))  # 第0天
                    close_1 = float(stock[i-2].get('close', 0))  # 第1天
                    close_2 = float(stock[i-1].get('close', 0))  # 第2天
                    close_3 = float(stock[i].get('close', 0))    # 第3天
                    
                    # 检查三连涨条件
                    if not (close_1 <= close_2 <= close_3):
                        continue
                        
                    # 检查V型底条件
                    window_start = max(0, i-3-window+1)
                    window_data = stock[window_start:i-2]
                    min_price = min(float(row.get('close', 0)) for row in window_data)
                    
                    if close_0 != min_price:
                        continue
                    
                    # 提取特征
                    feature_dict = {
                        'ts_code': ts_code,
                        'entry_date': stock[i].get('trade_date'),
                        'entry_price': close_3,
                        'close_0': close_0,
                        'close_1': close_1,
                        'close_2': close_2,
                        'close_3': close_3,
                    }
                    
                    features.append(feature_dict)
                except (ValueError, KeyError) as e:
                    continue
        
        print(f"特征提取完成，共 {len(features):,} 个信号")
        return features
    
    def backtest_signal(self, stock_data, signal_row):
        """回测单个信号"""
        ts_code = signal_row['ts_code']
        entry_date = signal_row['entry_date']
        entry_price = signal_row['entry_price']
        
        # 获取该股票在入场日期后的数据
        stock = [row for row in stock_data if row.get('ts_code') == ts_code]
        stock.sort(key=lambda x: x.get('trade_date', ''))
        
        # 找到入场日期在数据中的位置
        entry_idx = -1
        for i, row in enumerate(stock):
            if row.get('trade_date') == entry_date:
                entry_idx = i
                break
        
        if entry_idx == -1:
            return None
        
        # 初始化交易状态
        self.entry_price = entry_price
        self.remaining_shares = 1.0
        self.total_pnl = 0.0
        self.prev_close = entry_price
        self.is_first_drop = True
        self.stop_loss_price = entry_price * 0.969  # 3.1%止损
        
        # 从入场后第一天开始检查
        for i in range(entry_idx + 1, len(stock)):
            try:
                current_close = float(stock[i].get('close', 0))
                current_date = stock[i].get('trade_date')
                
                # 硬止损检查
                if current_close < self.stop_loss_price:
                    sell_pnl = self.remaining_shares * (current_close / self.entry_price - 1)
                    self.total_pnl += sell_pnl
                    return {
                        'ts_code': ts_code,
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_close,
                        'pnl': self.total_pnl,
                        'exit_reason': 'hard_stop_loss',
                        'holding_days': i - entry_idx
                    }
                
                # 分级减仓检查
                if current_close < self.prev_close:
                    if self.is_first_drop:
                        self.is_first_drop = False  # 豁免期
                    else:
                        # 计算跌幅
                        drop_pct = (self.prev_close - current_close) / self.prev_close
                        
                        if 0 < drop_pct <= 0.02:
                            sell_shares = self.remaining_shares * 0.10
                        elif drop_pct > 0.02:
                            sell_shares = self.remaining_shares * 0.20
                        else:
                            sell_shares = 0
                        
                        if sell_shares > 0:
                            sell_pnl = sell_shares * (current_close / self.entry_price - 1)
                            self.total_pnl += sell_pnl
                            self.remaining_shares -= sell_shares
                
                self.prev_close = current_close
                
                # 如果剩余份额太少，退出
                if self.remaining_shares < 0.01:
                    return {
                        'ts_code': ts_code,
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_close,
                        'pnl': self.total_pnl,
                        'exit_reason': 'partial_exit',
                        'holding_days': i - entry_idx
                    }
            except (ValueError, KeyError):
                continue
        
        # 如果持有到最后一天
        if len(stock) > entry_idx + 1:
            try:
                final_close = float(stock[-1].get('close', 0))
                final_date = stock[-1].get('trade_date')
                final_pnl = self.remaining_shares * (final_close / self.entry_price - 1)
                self.total_pnl += final_pnl
                
                return {
                    'ts_code': ts_code,
                    'entry_date': entry_date,
                    'exit_date': final_date,
                    'entry_price': entry_price,
                    'exit_price': final_close,
                    'pnl': self.total_pnl,
                    'exit_reason': 'end_of_data',
                    'holding_days': len(stock) - entry_idx - 1
                }
            except (ValueError, KeyError):
                pass
        
        return None
    
    def evaluate_features(self, features, stock_data):
        """评估特征组合的收益率"""
        print("开始评估特征组合...")
        
        results = []
        for signal in features:
            try:
                result = self.backtest_signal(stock_data, signal)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"回测信号时出错: {e}")
                continue
        
        if len(results) > 0:
            # 按收益率排序
            results.sort(key=lambda x: x['pnl'], reverse=True)
            
            print(f"回测完成，共 {len(results):,} 个交易")
            avg_pnl = sum(r['pnl'] for r in results) / len(results)
            print(f"平均收益率: {avg_pnl:.4f}")
            win_count = sum(1 for r in results if r['pnl'] > 0)
            win_rate = win_count / len(results)
            print(f"胜率: {win_rate:.2%}")
            max_pnl = max(r['pnl'] for r in results)
            min_pnl = min(r['pnl'] for r in results)
            print(f"最高收益率: {max_pnl:.4f}")
            print(f"最低收益率: {min_pnl:.4f}")
        
        return results
    
    def run_strategy(self, shanghai_file, shenzhen_file):
        """运行完整策略"""
        print("=" * 60)
        print("简化策略测试 - 计算深圳和上海收益率")
        print("=" * 60)
        
        # 加载数据
        shanghai_data = self.load_data(shanghai_file)
        shenzhen_data = self.load_data(shenzhen_file)
        
        if not shanghai_data and not shenzhen_data:
            print("无法加载数据文件，请检查文件路径")
            return
        
        # 合并数据
        all_data = shanghai_data + shenzhen_data
        print(f"合并后数据: {len(all_data):,} 行")
        
        # 提取特征
        features = self.extract_features(all_data)
        
        # 评估特征
        results = self.evaluate_features(features, all_data)
        
        # 分别计算上海和深圳的收益率
        shanghai_results = []
        shenzhen_results = []
        
        for result in results:
            ts_code = result['ts_code']
            if ts_code.startswith('00'):  # 深圳股票代码
                shenzhen_results.append(result)
            elif ts_code.startswith('60'):  # 上海股票代码
                shanghai_results.append(result)
        
        print("\n" + "=" * 60)
        print("分市场收益率统计")
        print("=" * 60)
        
        # 上海市场统计
        if shanghai_results:
            shanghai_avg = sum(r['pnl'] for r in shanghai_results) / len(shanghai_results)
            shanghai_win_rate = sum(1 for r in shanghai_results if r['pnl'] > 0) / len(shanghai_results)
            print(f"上海市场:")
            print(f"  交易次数: {len(shanghai_results):,}")
            print(f"  平均收益率: {shanghai_avg:.4f}")
            print(f"  胜率: {shanghai_win_rate:.2%}")
            print(f"  最高收益率: {max(r['pnl'] for r in shanghai_results):.4f}")
            print(f"  最低收益率: {min(r['pnl'] for r in shanghai_results):.4f}")
        else:
            print("上海市场: 无交易信号")
        
        print()
        
        # 深圳市场统计
        if shenzhen_results:
            shenzhen_avg = sum(r['pnl'] for r in shenzhen_results) / len(shenzhen_results)
            shenzhen_win_rate = sum(1 for r in shenzhen_results if r['pnl'] > 0) / len(shenzhen_results)
            print(f"深圳市场:")
            print(f"  交易次数: {len(shenzhen_results):,}")
            print(f"  平均收益率: {shenzhen_avg:.4f}")
            print(f"  胜率: {shenzhen_win_rate:.2%}")
            print(f"  最高收益率: {max(r['pnl'] for r in shenzhen_results):.4f}")
            print(f"  最低收益率: {min(r['pnl'] for r in shenzhen_results):.4f}")
        else:
            print("深圳市场: 无交易信号")
        
        return results

def main():
    """主函数"""
    strategy = SimpleStrategyTest()
    
    # 运行策略
    results = strategy.run_strategy(
        '上海测试数据.csv',
        '深圳测试数据.csv'
    )
    
    print("\n策略运行完成！")

if __name__ == "__main__":
    main()