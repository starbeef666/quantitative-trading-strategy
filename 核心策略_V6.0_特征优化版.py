#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心策略 V6.0 特征优化版
基于PRD要求：提取特征值，优化收益率，实现三连涨策略
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CoreStrategyV6:
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
        df = pd.read_csv(file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        print(f"数据加载完成，共 {len(df):,} 行")
        return df
    
    def extract_features(self, stock_data, window=20):
        """提取特征值"""
        features = []
        
        for ts_code in stock_data['ts_code'].unique():
            stock = stock_data[stock_data['ts_code'] == ts_code].copy()
            stock = stock.sort_values('trade_date').reset_index(drop=True)
            
            if len(stock) < window + 5:
                continue
                
            for i in range(window + 3, len(stock) - 2):
                # 基础特征
                close_0 = stock.iloc[i-3]['close']  # 第0天
                close_1 = stock.iloc[i-2]['close']  # 第1天
                close_2 = stock.iloc[i-1]['close']  # 第2天
                close_3 = stock.iloc[i]['close']    # 第3天
                
                # 检查三连涨条件
                if not (close_1 <= close_2 <= close_3):
                    continue
                    
                # 检查V型底条件
                window_start = max(0, i-3-window+1)
                window_data = stock.iloc[window_start:i-2]
                min_price = window_data['close'].min()
                
                if close_0 != min_price:
                    continue
                
                # 提取特征
                feature_dict = {
                    'ts_code': ts_code,
                    'entry_date': stock.iloc[i]['trade_date'],
                    'entry_price': close_3,
                    'close_0': close_0,
                    'close_1': close_1,
                    'close_2': close_2,
                    'close_3': close_3,
                    'volume_0': stock.iloc[i-3]['vol'],
                    'volume_1': stock.iloc[i-2]['vol'],
                    'volume_2': stock.iloc[i-1]['vol'],
                    'volume_3': stock.iloc[i]['vol'],
                    'amount_0': stock.iloc[i-3]['amount'],
                    'amount_1': stock.iloc[i-2]['amount'],
                    'amount_2': stock.iloc[i-1]['amount'],
                    'amount_3': stock.iloc[i]['amount'],
                    'pct_chg_0': stock.iloc[i-3]['pct_chg'],
                    'pct_chg_1': stock.iloc[i-2]['pct_chg'],
                    'pct_chg_2': stock.iloc[i-1]['pct_chg'],
                    'pct_chg_3': stock.iloc[i]['pct_chg'],
                    'high_0': stock.iloc[i-3]['high'],
                    'high_1': stock.iloc[i-2]['high'],
                    'high_2': stock.iloc[i-1]['high'],
                    'high_3': stock.iloc[i]['high'],
                    'low_0': stock.iloc[i-3]['low'],
                    'low_1': stock.iloc[i-2]['low'],
                    'low_2': stock.iloc[i-1]['low'],
                    'low_3': stock.iloc[i]['low'],
                    'open_0': stock.iloc[i-3]['open'],
                    'open_1': stock.iloc[i-2]['open'],
                    'open_2': stock.iloc[i-1]['open'],
                    'open_3': stock.iloc[i]['open'],
                }
                
                # 计算衍生特征
                feature_dict.update({
                    'price_momentum': (close_3 - close_0) / close_0,
                    'volume_momentum': (feature_dict['volume_3'] - feature_dict['volume_0']) / feature_dict['volume_0'],
                    'volatility': (feature_dict['high_3'] - feature_dict['low_3']) / feature_dict['open_3'],
                    'avg_volume': (feature_dict['volume_1'] + feature_dict['volume_2'] + feature_dict['volume_3']) / 3,
                    'avg_amount': (feature_dict['amount_1'] + feature_dict['amount_2'] + feature_dict['amount_3']) / 3,
                    'avg_pct_chg': (feature_dict['pct_chg_1'] + feature_dict['pct_chg_2'] + feature_dict['pct_chg_3']) / 3,
                })
                
                features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        print(f"特征提取完成，共 {len(features_df):,} 个信号")
        return features_df
    
    def backtest_signal(self, stock_data, signal_row):
        """回测单个信号"""
        ts_code = signal_row['ts_code']
        entry_date = signal_row['entry_date']
        entry_price = signal_row['entry_price']
        
        # 获取该股票在入场日期后的数据
        stock = stock_data[stock_data['ts_code'] == ts_code].copy()
        stock = stock.sort_values('trade_date').reset_index(drop=True)
        
        # 找到入场日期在数据中的位置
        entry_idx = stock[stock['trade_date'] == entry_date].index[0]
        
        # 初始化交易状态
        self.entry_price = entry_price
        self.remaining_shares = 1.0
        self.total_pnl = 0.0
        self.prev_close = entry_price
        self.is_first_drop = True
        self.stop_loss_price = entry_price * 0.969  # 3.1%止损
        
        # 从入场后第一天开始检查
        for i in range(entry_idx + 1, len(stock)):
            current_close = stock.iloc[i]['close']
            current_date = stock.iloc[i]['trade_date']
            
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
        
        # 如果持有到最后一天
        final_pnl = self.remaining_shares * (current_close / self.entry_price - 1)
        self.total_pnl += final_pnl
        
        return {
            'ts_code': ts_code,
            'entry_date': entry_date,
            'exit_date': current_date,
            'entry_price': entry_price,
            'exit_price': current_close,
            'pnl': self.total_pnl,
            'exit_reason': 'end_of_data',
            'holding_days': len(stock) - entry_idx - 1
        }
    
    def evaluate_features(self, features_df, stock_data):
        """评估特征组合的收益率"""
        print("开始评估特征组合...")
        
        results = []
        for idx, signal in features_df.iterrows():
            try:
                result = self.backtest_signal(stock_data, signal)
                result.update({
                    'price_momentum': signal['price_momentum'],
                    'volume_momentum': signal['volume_momentum'],
                    'volatility': signal['volatility'],
                    'avg_volume': signal['avg_volume'],
                    'avg_amount': signal['avg_amount'],
                    'avg_pct_chg': signal['avg_pct_chg'],
                })
                results.append(result)
            except Exception as e:
                print(f"回测信号 {idx} 时出错: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # 按收益率排序
            results_df = results_df.sort_values('pnl', ascending=False)
            
            print(f"回测完成，共 {len(results_df):,} 个交易")
            print(f"平均收益率: {results_df['pnl'].mean():.4f}")
            print(f"胜率: {(results_df['pnl'] > 0).mean():.2%}")
            print(f"最高收益率: {results_df['pnl'].max():.4f}")
            print(f"最低收益率: {results_df['pnl'].min():.4f}")
        
        return results_df
    
    def optimize_features(self, results_df):
        """基于收益率优化特征组合"""
        print("开始特征优化...")
        
        # 选择收益率最高的前20%作为优秀样本
        top_percentile = 0.2
        top_count = int(len(results_df) * top_percentile)
        top_results = results_df.head(top_count)
        
        # 计算优秀样本的特征统计
        feature_stats = {}
        feature_columns = ['price_momentum', 'volume_momentum', 'volatility', 
                          'avg_volume', 'avg_amount', 'avg_pct_chg']
        
        for feature in feature_columns:
            feature_stats[feature] = {
                'mean': top_results[feature].mean(),
                'std': top_results[feature].std(),
                'min': top_results[feature].min(),
                'max': top_results[feature].max()
            }
        
        print("优秀样本特征统计:")
        for feature, stats in feature_stats.items():
            print(f"{feature}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
        
        return feature_stats
    
    def run_strategy(self, shanghai_file, shenzhen_file):
        """运行完整策略"""
        print("=" * 60)
        print("核心策略 V6.0 特征优化版")
        print("=" * 60)
        
        # 加载数据
        shanghai_data = self.load_data(shanghai_file)
        shenzhen_data = self.load_data(shenzhen_file)
        
        # 合并数据
        all_data = pd.concat([shanghai_data, shenzhen_data], ignore_index=True)
        print(f"合并后数据: {len(all_data):,} 行")
        
        # 提取特征
        features_df = self.extract_features(all_data)
        
        # 评估特征
        results_df = self.evaluate_features(features_df, all_data)
        
        # 优化特征
        if len(results_df) > 0:
            feature_stats = self.optimize_features(results_df)
            
            # 保存结果
            results_df.to_csv('核心策略_V6.0_回测结果.csv', index=False)
            print("结果已保存到: 核心策略_V6.0_回测结果.csv")
        
        return results_df

def main():
    """主函数"""
    strategy = CoreStrategyV6()
    
    # 运行策略
    results = strategy.run_strategy(
        '上海主板_历史数据_2018至今_20250729_233546_副本2.csv',
        '深圳主板_历史数据_2018至今_20250729_233546_副本.csv'
    )
    
    print("\n策略运行完成！")

if __name__ == "__main__":
    main() 