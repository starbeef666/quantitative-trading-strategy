#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略收益率分析 - 详细分析深圳和上海的收益率
"""

import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StrategyAnalyzer:
    def __init__(self):
        """初始化分析器"""
        self.entry_price = None
        self.remaining_shares = 1.0
        self.total_pnl = 0.0
        self.prev_close = None
        self.is_first_drop = True
        self.stop_loss_price = None
        
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
    
    def analyze_results(self, results):
        """详细分析结果"""
        if not results:
            print("没有交易结果可分析")
            return
        
        # 总体统计
        total_trades = len(results)
        total_pnl = sum(r['pnl'] for r in results)
        avg_pnl = total_pnl / total_trades
        win_trades = [r for r in results if r['pnl'] > 0]
        lose_trades = [r for r in results if r['pnl'] < 0]
        win_rate = len(win_trades) / total_trades
        
        print("\n" + "=" * 80)
        print("策略收益率详细分析")
        print("=" * 80)
        
        print(f"总体统计:")
        print(f"  总交易次数: {total_trades:,}")
        print(f"  总收益率: {total_pnl:.4f}")
        print(f"  平均收益率: {avg_pnl:.4f}")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  盈利交易: {len(win_trades):,}")
        print(f"  亏损交易: {len(lose_trades):,}")
        
        if win_trades:
            avg_win = sum(r['pnl'] for r in win_trades) / len(win_trades)
            max_win = max(r['pnl'] for r in win_trades)
            print(f"  平均盈利: {avg_win:.4f}")
            print(f"  最大盈利: {max_win:.4f}")
        
        if lose_trades:
            avg_loss = sum(r['pnl'] for r in lose_trades) / len(lose_trades)
            max_loss = min(r['pnl'] for r in lose_trades)
            print(f"  平均亏损: {avg_loss:.4f}")
            print(f"  最大亏损: {max_loss:.4f}")
        
        # 按退出原因分析
        exit_reasons = {}
        for r in results:
            reason = r['exit_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = []
            exit_reasons[reason].append(r)
        
        print(f"\n退出原因分析:")
        for reason, trades in exit_reasons.items():
            avg_pnl_reason = sum(r['pnl'] for r in trades) / len(trades)
            win_rate_reason = sum(1 for r in trades if r['pnl'] > 0) / len(trades)
            print(f"  {reason}: {len(trades):,} 次, 平均收益率 {avg_pnl_reason:.4f}, 胜率 {win_rate_reason:.2%}")
        
        # 按持有天数分析
        holding_days = [r['holding_days'] for r in results]
        avg_holding = sum(holding_days) / len(holding_days)
        max_holding = max(holding_days)
        min_holding = min(holding_days)
        print(f"\n持有天数分析:")
        print(f"  平均持有天数: {avg_holding:.1f}")
        print(f"  最长持有天数: {max_holding}")
        print(f"  最短持有天数: {min_holding}")
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'win_trades': len(win_trades),
            'lose_trades': len(lose_trades)
        }
    
    def run_analysis(self, shanghai_file, shenzhen_file):
        """运行完整分析"""
        print("=" * 80)
        print("策略收益率分析")
        print("=" * 80)
        
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
        
        # 回测所有信号
        print("开始回测...")
        results = []
        for signal in features:
            try:
                result = self.backtest_signal(all_data, signal)
                if result:
                    results.append(result)
            except Exception as e:
                continue
        
        print(f"回测完成，共 {len(results):,} 个交易")
        
        # 分别分析上海和深圳
        shanghai_results = []
        shenzhen_results = []
        
        for result in results:
            ts_code = result['ts_code']
            if ts_code.startswith('60'):  # 上海股票代码
                shanghai_results.append(result)
            elif ts_code.startswith('00'):  # 深圳股票代码
                shenzhen_results.append(result)
        
        # 总体分析
        print("\n总体分析:")
        overall_stats = self.analyze_results(results)
        
        # 上海市场分析
        print(f"\n上海市场分析:")
        if shanghai_results:
            shanghai_stats = self.analyze_results(shanghai_results)
        else:
            print("  无交易信号")
        
        # 深圳市场分析
        print(f"\n深圳市场分析:")
        if shenzhen_results:
            shenzhen_stats = self.analyze_results(shenzhen_results)
        else:
            print("  无交易信号")
        
        # 保存详细结果
        if results:
            with open('策略分析结果.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ts_code', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl', 'exit_reason', 'holding_days'])
                for result in results:
                    writer.writerow([
                        result['ts_code'], result['entry_date'], result['exit_date'],
                        result['entry_price'], result['exit_price'], result['pnl'],
                        result['exit_reason'], result['holding_days']
                    ])
            print(f"\n详细结果已保存到: 策略分析结果.csv")
        
        return results

def main():
    """主函数"""
    analyzer = StrategyAnalyzer()
    
    # 运行分析
    results = analyzer.run_analysis(
        '上海测试数据.csv',
        '深圳测试数据.csv'
    )
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()