import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# 核心功能函数 (源自 终极策略_V28.0_最终版.txt)
# ==============================================================================

def check_entry_conditions(data, i, n=20):
    """检查V6入场信号，确保无重复和高起点。"""
    if i < 4 + n: return False
    
    # 连续4天不减
    if not (data['close'].iloc[i-3] <= data['close'].iloc[i-2] <= data['close'].iloc[i-1] <= data['close'].iloc[i]):
        return False
    
    # V型底
    zero_day = i - 4
    window_start = zero_day - n + 1
    if window_start < 0: return False
    if data['close'].iloc[zero_day] != data['close'].iloc[window_start : zero_day + 1].min():
        return False
    
    return True

def extract_features(data, signals_df):
    """为信号提取13个核心特征"""
    features_list = []
    # 确保列是数值类型
    for col in ['close', 'high', 'low', 'open', 'vol', 'amount']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    for idx, signal in signals_df.iterrows():
        i = signal['entry_day_idx']
        
        if i < 30: continue

        close = data['close'].iloc[:i+1]
        high = data['high'].iloc[:i+1]
        low = data['low'].iloc[:i+1]
        open_price = data['open'].iloc[:i+1]
        volume = data['vol'].iloc[:i+1]
        amount = data['amount'].iloc[:i+1]

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]

        gain = (delta.where(delta > 0, 0)).rolling(window=30, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=30, min_periods=1).mean()
        rs = gain / loss
        rsi_30 = 100 - (100 / (1 + rs)).iloc[-1]
        
        tr1 = high.iloc[-20:].max() - low.iloc[-20:].min()
        tr2 = abs(high.iloc[-20:].max() - close.iloc[-21]) if i > 20 else 0
        tr3 = abs(low.iloc[-20:].min() - close.iloc[-21]) if i > 20 else 0
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().iloc[-1]
        atr_ratio = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0
        
        body = abs(close.iloc[-1] - open_price.iloc[-1])
        full_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = body / full_range if full_range > 0 else 0
        upper_shadow_ratio = (high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])) / full_range if full_range > 0 else 0
        lower_shadow_ratio = (min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]) / full_range if full_range > 0 else 0

        vol_mean_20 = volume.iloc[-21:-1].mean()
        vol_amp = (volume.iloc[-1] / vol_mean_20) -1 if vol_mean_20 > 0 else 0
        amount_mean_20 = amount.iloc[-21:-1].mean()
        amount_amp = (amount.iloc[-1] / amount_mean_20) - 1 if amount_mean_20 > 0 else 0

        ma5 = close.rolling(window=5).mean().iloc[-1]
        ma10 = close.rolling(window=10).mean().iloc[-1]
        ma5_above_ma10 = 1 if ma5 > ma10 else 0
        close_above_ma5 = 1 if close.iloc[-1] > ma5 else 0

        gap_open = (open_price.iloc[-1] / close.iloc[-2]) - 1 if i > 0 and close.iloc[-2] > 0 else 0
        consecutive_up_days = signal['consecutive_up_days']
        
        ma20 = close.rolling(window=20).mean().iloc[i-4] if i-4 >= 20 else close.iloc[i-4]
        bottom_depth = (ma20 - close.iloc[i-4]) / close.iloc[i-4] if close.iloc[i-4] > 0 else 0

        features_list.append({
            'signal_index': idx,
            'rsi_14': rsi_14, 'rsi_30': rsi_30, 'atr_ratio': atr_ratio, 'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio, 'lower_shadow_ratio': lower_shadow_ratio,
            'vol_amp': vol_amp, 'amount_amp': amount_amp, 'ma5_above_ma10': ma5_above_ma10,
            'close_above_ma5': close_above_ma5, 'gap_open': gap_open, 
            'bottom_depth': bottom_depth
        })
        
    features_df = pd.DataFrame(features_list)
    return features_df.fillna(0)

def simulate_trade(data, entry_idx, entry_price, holding_period=10):
    """模拟单笔交易的风控与收益计算，确保分段累加。"""
    remaining = 1.0
    total_pnl = 0.0
    prev = entry_price
    first_drop = True
    stop_loss = entry_price * (1 - 0.031)

    exit_day = min(entry_idx + holding_period, len(data) - 1)
    
    for day in range(entry_idx + 1, exit_day + 1):
        close = data['close'].iloc[day]
        
        if close < stop_loss:
            total_pnl += remaining * (close / entry_price - 1)
            remaining = 0
            break
        
        if close < prev:
            if first_drop:
                first_drop = False
            else:
                drop_pct = (prev - close) / prev
                sell_pct = 0.1 if drop_pct <= 0.02 else 0.2
                sell_shares = remaining * sell_pct
                total_pnl += sell_shares * (close / entry_price - 1)
                remaining -= sell_shares
        
        prev = close
    
    if remaining > 0:
        exit_price = data['close'].iloc[exit_day]
        total_pnl += remaining * (exit_price / entry_price - 1)
        
    return total_pnl

def run_backtest(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return

    print("Data loaded. Finding signals...")
    all_signals = []
    for ts_code, stock_data in tqdm(df.groupby('ts_code'), desc="Scanning Stocks"):
        stock_data = stock_data.sort_values('trade_date').reset_index(drop=True)
        if len(stock_data) < 25: continue
        
        for i in range(24, len(stock_data)):
            if check_entry_conditions(stock_data, i):
                all_signals.append({
                    'ts_code': ts_code,
                    'trade_date': stock_data.at[i, 'trade_date'],
                    'entry_day_idx': i,
                    'entry_price': stock_data.at[i, 'close'],
                    'consecutive_up_days': 4 # V6固定为4
                })

    if not all_signals:
        print("未找到任何交易信号。")
        return
        
    signals_df = pd.DataFrame(all_signals).reset_index().rename(columns={'index':'original_index'})
    print(f"\nFound {len(signals_df)} total signals. Extracting features...")

    all_features = []
    for ts_code, group_signals in tqdm(signals_df.groupby('ts_code'), desc="Extracting Features"):
        stock_data = df[df['ts_code'] == ts_code].sort_values('trade_date').reset_index(drop=True)
        features = extract_features(stock_data, group_signals)
        all_features.append(features)
    
    if not all_features:
        print("特征提取失败，未生成任何特征。")
        return

    # 过滤掉空的特征集
    all_features = [f for f in all_features if not f.empty]
    if not all_features:
        print("所有特征提取都失败了。")
        return
    
    # 过滤掉空的特征集
    all_features = [f for f in all_features if not f.empty]
    if not all_features:
        print("所有股票的特征提取都失败了。")
        return

    features_df = pd.concat(all_features, ignore_index=True)
    
    # 关键修正：使用original_index 和 signal_index 进行合并，更稳健
    signals_df = pd.merge(signals_df, features_df, left_on='original_index', right_on='signal_index')
    signals_df = signals_df.drop(columns=['signal_index']) # 清理多余的列
    
    print("Features extracted. Scoring signals with AI model...")
    try:
        model = joblib.load('v27_universal_model.pkl')
    except FileNotFoundError:
        print("错误: 模型文件 'v27_universal_model.pkl' 未找到。请确保它在脚本所在目录。")
        return
        
    feature_names = model.feature_name_
    features_df = signals_df[feature_names]
    signals_df['评分'] = model.predict(features_df)

    print("Scoring complete. Assigning positions...")
    top_percentile = 0.4
    threshold = signals_df['评分'].quantile(1 - top_percentile)
    elite_signals = signals_df[signals_df['评分'] >= threshold].copy()
    
    elite_signals['rank_pct'] = elite_signals['评分'].rank(pct=True) * 100
    pos_range = 15.0 - 2.0
    elite_signals['position'] = 2.0 + (elite_signals['rank_pct'] / 100) * pos_range

    print(f"Simulating trades for {len(elite_signals)} elite signals...")
    pnls = []
    for idx, signal in tqdm(elite_signals.iterrows(), total=len(elite_signals), desc="Simulating Trades"):
        stock_data = df[df['ts_code'] == signal['ts_code']].sort_values('trade_date').reset_index(drop=True)
        pnl = simulate_trade(stock_data, signal['entry_day_idx'], signal['entry_price'])
        final_pnl = pnl * (signal['position'] / 100)
        pnls.append(final_pnl)
    
    elite_signals['pnl'] = pnls

    print("\n--- Backtest Results ---")
    
    def analyze_performance(df, description):
        if df.empty:
            print(f"\n{description}: No trades to analyze.")
            return
        win_rate = (df['pnl'] > 0).mean() * 100
        expected_return = df['pnl'].mean() * 100 
        
        print(f"\n{description}:")
        print(f"  交易总数: {len(df)}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  期望收益: {expected_return:.4f}%")
        print("--------------------")

    analyze_performance(elite_signals, "Top 40% Elite Signals")
    
    top_20_threshold = elite_signals['评分'].quantile(0.5)
    top_20_signals = elite_signals[elite_signals['评分'] >= top_20_threshold]
    analyze_performance(top_20_signals, "Top 20% Elite Signals")


if __name__ == '__main__':
    DATA_FILE = "/Users/yamijin/Desktop/A股主板历史数据_2018至今_20250729_233546/上海主板_历史数据_2018至今_20250729_233546_副本2.csv"
    run_backtest(DATA_FILE)
