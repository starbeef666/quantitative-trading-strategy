import pandas as pd
import numpy as np
import time
import warnings
import os

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (V28.0 策略 - V6+AI筛选版)
# ==============================================================================
DATA_PATH = '/Users/yamijin/Desktop/A股主板历史数据_2018至今_20250729_233546/上海主板_历史数据_2018至今_20250729_233546_副本2.csv'
OUTPUT_CSV_PATH = 'V26_V28_Enhanced_Trades.csv'
MODEL_PATH = 'v27_universal_model.pkl'  # AI模型路径

# V28 核心参数
V_BOTTOM_LOOKBACK = 20  # V型底回看天数 (n≥20)
STOP_LOSS_PCT = 0.031   # 固定止损比例
MAX_HOLDING_DAYS = 10   # 最大持仓天数
TOP_PERCENTILE = 0.4    # Top 40%筛选
MIN_POSITION = 2.0      # 最小仓位%
MAX_POSITION = 15.0     # 最大仓位%

# ==============================================================================
# 数据加载与预处理
# ==============================================================================
def load_and_preprocess_data(file_path):
    """加载数据并进行基础预处理"""
    print(f"开始加载数据...")
    start_time = time.time()
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载 {len(df):,} 条数据记录")
    except FileNotFoundError:
        print(f"错误：数据文件未找到！")
        return None

    df.rename(columns={
        'ts_code': '代码', 'trade_date': '日期', 'open': '开盘',
        'high': '最高', 'low': '最低', 'close': '收盘',
        'vol': '成交量', 'amount': '成交额'
    }, inplace=True)
    
    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
    df.sort_values(by=['代码', '日期'], inplace=True)
    
    # 计算技术指标用于特征提取
    grouped = df.groupby('代码')
    df['ma5'] = grouped['收盘'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = grouped['收盘'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = grouped['收盘'].transform(lambda x: x.rolling(20).mean())
    df['vol_ma20'] = grouped['成交量'].transform(lambda x: x.rolling(20).mean())
    
    # 计算RSI
    def calculate_rsi(prices, window):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = grouped['收盘'].transform(lambda x: calculate_rsi(x, 14))
    df['rsi_30'] = grouped['收盘'].transform(lambda x: calculate_rsi(x, 30))
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f} 秒")
    return df

# ==============================================================================
# V6核心信号检测 (复用之前的逻辑)
# ==============================================================================
def find_v6_base_signals(all_data):
    """寻找所有符合V6.0核心入场条件的交易信号"""
    print("开始寻找V6.0核心信号...")
    start_time = time.time()
    
    signals = []
    grouped = all_data.groupby('代码')
    
    for code, stock_data in grouped:
        if len(stock_data) < V_BOTTOM_LOOKBACK + 5:
            continue
            
        df = stock_data.copy().reset_index(drop=True)
        closes = df['收盘'].values
        
        for i in range(V_BOTTOM_LOOKBACK + 4, len(closes)):
            # 条件1: 连续4天收盘价不减 A≤B≤C≤D
            if not (closes[i-3] <= closes[i-2] <= closes[i-1] <= closes[i]):
                continue
                
            # 条件2: V型底 - 第0天(i-4)是过去n天最低点
            day0_idx = i - 4
            window_start = max(0, day0_idx - V_BOTTOM_LOOKBACK + 1)
            window_closes = closes[window_start:day0_idx + 1]
            
            if closes[day0_idx] != min(window_closes):
                continue
                
            # 记录信号和对应的数据索引
            signals.append({
                '日期': df.iloc[i]['日期'],
                '代码': code,
                '买入价格': closes[i],
                '数据索引': i,  # 用于后续特征提取
                '股票数据': df   # 保存完整数据用于特征计算
            })
            
    print(f"V6信号寻找完成，共找到 {len(signals)} 个，耗时: {time.time() - start_time:.2f} 秒")
    return signals

# ==============================================================================
# V28特征提取 (13个维度)
# ==============================================================================
def extract_v28_features(signal_data, signal_idx):
    """提取V28策略的13个特征"""
    try:
        if signal_idx < 30:  # 确保有足够历史数据
            return None
            
        current = signal_data.iloc[signal_idx]
        
        # 基础数据
        open_price = current['开盘']
        high_price = current['最高']
        low_price = current['最低']
        close_price = current['收盘']
        volume = current['成交量']
        
        # 1-2. RSI指标
        rsi_14 = current.get('rsi_14', 50)
        rsi_30 = current.get('rsi_30', 50)
        
        # 3. ATR比率
        high_low = high_price - low_price
        recent_ranges = []
        for j in range(max(0, signal_idx-13), signal_idx+1):
            if j < len(signal_data):
                recent_ranges.append(signal_data.iloc[j]['最高'] - signal_data.iloc[j]['最低'])
        avg_range = np.mean(recent_ranges) if recent_ranges else high_low
        atr_ratio = high_low / avg_range if avg_range > 0 else 1
        
        # 4-6. K线形态特征
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        body_ratio = body_size / total_range if total_range > 0 else 0
        
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
        lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else 0
        
        # 7-8. 成交量特征
        vol_ma20 = current.get('vol_ma20', volume)
        vol_amp = volume / vol_ma20 if vol_ma20 > 0 else 1
        
        # 假设成交额放大倍数
        amount_amp = 1.5  # 简化处理
        
        # 9-10. 均线关系
        ma5 = current.get('ma5', close_price)
        ma10 = current.get('ma10', close_price)
        ma5_above_ma10 = 1 if ma5 > ma10 else 0
        close_above_ma5 = 1 if close_price > ma5 else 0
        
        # 11. 跳空开盘
        gap_open = 0
        if signal_idx > 0:
            prev_close = signal_data.iloc[signal_idx-1]['收盘']
            gap_open = 1 if open_price > prev_close * 1.02 else 0
        
        # 12. 连续上涨天数
        consecutive_up_days = 0
        for j in range(signal_idx, max(0, signal_idx-10), -1):
            if j > 0:
                if signal_data.iloc[j]['收盘'] > signal_data.iloc[j-1]['收盘']:
                    consecutive_up_days += 1
                else:
                    break
        
        # 13. 底部深度 (V28核心特征)
        ma20 = current.get('ma20', close_price)
        day0_close = signal_data.iloc[signal_idx-4]['收盘']  # 第0天收盘价
        bottom_depth = (ma20 - day0_close) / ma20 * 100 if ma20 > 0 else 0
        
        features = {
            'rsi_14': rsi_14,
            'rsi_30': rsi_30,
            'atr_ratio': atr_ratio,
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'vol_amp': vol_amp,
            'amount_amp': amount_amp,
            'ma5_above_ma10': ma5_above_ma10,
            'close_above_ma5': close_above_ma5,
            'gap_open': gap_open,
            'consecutive_up_days': consecutive_up_days,
            'bottom_depth': bottom_depth
        }
        
        return features
        
    except Exception as e:
        print(f"特征提取错误: {e}")
        return None

# ==============================================================================
# V28简化评分模型 (无AI模型时的备选)
# ==============================================================================
def v28_scoring_model(features):
    """V28增强评分模型，基于V28文档的关键特征"""
    if not features:
        return 0
    
    score = 0
    
    # RSI超卖奖励 (更严格)
    if features['rsi_14'] < 25:
        score += 0.4
    elif features['rsi_14'] < 35:
        score += 0.2
    elif features['rsi_14'] < 50:
        score += 0.1
    
    # 底部深度奖励 (V28核心)
    if features['bottom_depth'] > 15:
        score += 0.3
    elif features['bottom_depth'] > 8:
        score += 0.2
    elif features['bottom_depth'] > 3:
        score += 0.1
    
    # 成交量放大奖励
    if features['vol_amp'] > 3:
        score += 0.25
    elif features['vol_amp'] > 2:
        score += 0.15
    elif features['vol_amp'] > 1.5:
        score += 0.1
    
    # 均线关系奖励
    score += features['ma5_above_ma10'] * 0.15
    score += features['close_above_ma5'] * 0.1
    
    # 连续上涨奖励
    if features['consecutive_up_days'] >= 4:
        score += 0.2
    elif features['consecutive_up_days'] >= 2:
        score += 0.1
    
    # K线形态奖励
    if 0.4 < features['body_ratio'] < 0.8:
        score += 0.1
    
    # ATR奖励 (波动性)
    if 1.2 < features['atr_ratio'] < 2.0:
        score += 0.05
    
    return score

# ==============================================================================
# V28精英筛选与仓位分配
# ==============================================================================
def assign_v28_positions(signals_df, top_percentile=TOP_PERCENTILE, min_pos=MIN_POSITION, max_pos=MAX_POSITION):
    """V28精英筛选和线性仓位分配"""
    print(f"开始V28精英筛选 (Top {top_percentile*100:.0f}%)...")
    
    if len(signals_df) < 100:
        print("警告：信号数量过少，可能影响流动性")
    
    # 筛选Top 40%
    threshold = signals_df['AI评分'].quantile(1 - top_percentile)
    elite_signals = signals_df[signals_df['AI评分'] >= threshold].copy()
    
    # 在Top 40%内进行排名
    elite_signals['rank_pct'] = elite_signals['AI评分'].rank(pct=True) * 100
    
    # 线性映射仓位: 2% 到 15%
    pos_range = max_pos - min_pos
    elite_signals['仓位'] = min_pos + (elite_signals['rank_pct'] / 100) * pos_range
    
    print(f"筛选完成：从 {len(signals_df)} 个信号中选出 {len(elite_signals)} 个精英信号")
    print(f"评分阈值: {threshold:.4f}")
    
    return elite_signals

# ==============================================================================
# V28回测逻辑 (V6风控 + V28仓位)
# ==============================================================================
def run_v28_backtest(all_data, elite_signals):
    """执行V28策略回测，结合V6风控和V28仓位管理"""
    print("开始执行V28策略回测...")
    start_time = time.time()
    
    trades = []
    active_trades = {}
    
    if elite_signals.empty:
        return pd.DataFrame()
        
    unique_dates = sorted(elite_signals['日期'].unique())
    stock_groups = all_data.groupby('代码')

    for current_date in unique_dates:
        # 清理已到期的持仓
        ended_stocks = [code for code, end_date in active_trades.items() if current_date >= end_date]
        for code in ended_stocks:
            del active_trades[code]
        
        daily_signals = elite_signals[elite_signals['日期'] == current_date]
        
        for _, signal in daily_signals.iterrows():
            stock_code = signal['代码']
            
            if stock_code in active_trades:
                continue
            
            entry_price = signal['买入价格']
            buy_date = signal['日期']
            position_size = signal['仓位'] / 100  # 转换为小数
            
            # 获取该股票的未来数据
            if stock_code not in stock_groups.groups:
                continue
                
            stock_data = stock_groups.get_group(stock_code)
            future_data = stock_data[stock_data['日期'] > buy_date].head(MAX_HOLDING_DAYS + 2)
            
            if len(future_data) < 1:
                continue

            # V6交易管理 + V28仓位调整
            remaining_shares = position_size  # 初始仓位根据AI评分调整
            total_pnl = 0.0
            prev_close = entry_price
            is_first_drop = True
            
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            
            sell_date = None
            exit_reason = f"持有{MAX_HOLDING_DAYS}天到期"

            for i in range(min(len(future_data), MAX_HOLDING_DAYS)):
                current_row = future_data.iloc[i]
                current_close = current_row['收盘']
                
                # 1. 硬止损检查
                if current_close < stop_loss_price:
                    sell_date = current_row['日期']
                    remaining_pnl = remaining_shares * (stop_loss_price / entry_price - 1)
                    total_pnl += remaining_pnl
                    remaining_shares = 0
                    exit_reason = "硬止损"
                    break

                # 2. 分级减仓与豁免期
                if current_close < prev_close:
                    if is_first_drop:
                        is_first_drop = False
                    else:
                        drop_pct = (prev_close - current_close) / prev_close
                        sell_shares = 0
                        if 0 < drop_pct <= 0.02:
                            sell_shares = remaining_shares * 0.10
                        elif drop_pct > 0.02:
                            sell_shares = remaining_shares * 0.20
                        
                        if sell_shares > 0:
                            sell_pnl = sell_shares * (current_close / entry_price - 1)
                            total_pnl += sell_pnl
                            remaining_shares -= sell_shares
                
                prev_close = current_close

            # 3. 最终退出处理
            if sell_date is None:
                exit_day_index = min(MAX_HOLDING_DAYS - 1, len(future_data) - 1)
                sell_date = future_data.iloc[exit_day_index]['日期']
                exit_price = future_data.iloc[exit_day_index]['收盘']
                if remaining_shares > 0:
                    remaining_pnl = remaining_shares * (exit_price / entry_price - 1)
                    total_pnl += remaining_pnl
            
            trades.append({
                '代码': stock_code,
                '买入日期': buy_date,
                '买入价格': entry_price,
                '卖出日期': sell_date,
                '总收益率': total_pnl,
                '退出原因': exit_reason,
                'AI评分': signal['AI评分'],
                '初始仓位': signal['仓位']
            })
            
            active_trades[stock_code] = sell_date

    print(f"V28回测完成，耗时: {time.time() - start_time:.2f} 秒")
    return pd.DataFrame(trades)

# ==============================================================================
# 性能分析
# ==============================================================================
def analyze_v28_performance(trades_df):
    if trades_df.empty:
        print("V28回测未产生任何交易。")
        return

    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['总收益率'] > 0]
    loss_trades = trades_df[trades_df['总收益率'] <= 0]
    
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    avg_return = trades_df['总收益率'].mean()
    avg_win = win_trades['总收益率'].mean() if not win_trades.empty else 0
    avg_loss = loss_trades['总收益率'].mean() if not loss_trades.empty else 0
    
    print("\n" + "="*60)
    print("V28.0 策略性能分析 (V6+AI筛选增强版)")
    print("="*60)
    print(f"总交易次数: {total_trades:,.0f}")
    print(f"胜率: {win_rate:.2%}")
    print(f"期望收益 (平均总收益率): {avg_return:.4%}")
    print(f"平均盈利收益率: {avg_win:.4%}")
    print(f"平均亏损收益率: {avg_loss:.4%}")
    if avg_loss != 0:
        print(f"盈亏比: {-avg_win / avg_loss:.2f}")
    
    # 与V6和V28预期对比
    print(f"\n--- 与基准对比 ---")
    print(f"V6基准期望收益: 1.57%")
    print(f"V28实际期望收益: {avg_return:.4%}")
    print(f"V28预期目标: >1.08% (PRD要求)")
    
    if avg_return >= 0.02:
        print("✅ 期望收益达到2%目标！")
    elif avg_return > 0.0157:
        print("✅ 期望收益超越V6基准！")
    elif avg_return > 0.0108:
        print("✅ 期望收益达到V28 PRD目标！")
    else:
        print("❌ 期望收益未达到PRD目标")
    
    print(f"\n交易记录已保存至: '{OUTPUT_CSV_PATH}'")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    # 1. 加载数据
    all_data = load_and_preprocess_data(DATA_PATH)
    if all_data is None:
        return
        
    # 2. 寻找V6基础信号
    v6_signals = find_v6_base_signals(all_data)
    
    if not v6_signals:
        print("未找到任何V6信号，无法进行V28筛选。")
        return
    
    # 3. 提取特征并评分
    print("开始提取V28特征并评分...")
    scored_signals = []
    
    for signal in v6_signals:
        features = extract_v28_features(signal['股票数据'], signal['数据索引'])
        if features:
            score = v28_scoring_model(features)
            scored_signals.append({
                '日期': signal['日期'],
                '代码': signal['代码'],
                '买入价格': signal['买入价格'],
                'AI评分': score
            })
    
    signals_df = pd.DataFrame(scored_signals)
    print(f"成功为 {len(signals_df)} 个信号完成特征提取和评分")
    
    if signals_df.empty:
        print("特征提取失败，无法进行V28筛选。")
        return
    
    # 4. V28精英筛选和仓位分配
    elite_signals = assign_v28_positions(signals_df)
    
    if elite_signals.empty:
        print("精英筛选后无信号，无法进行回测。")
        return
    
    # 5. 执行V28回测
    trades_df = run_v28_backtest(all_data, elite_signals)
    
    # 6. 保存和分析结果
    if not trades_df.empty:
        trades_df.sort_values(by='买入日期', inplace=True)
        trades_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        analyze_v28_performance(trades_df)

if __name__ == '__main__':
    main()