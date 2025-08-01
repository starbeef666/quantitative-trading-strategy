import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (V32 AI精英策略)
# ==============================================================================
DATA_PATH = '/Users/yamijin/Desktop/A股主板历史数据_2018至今_20250729_233546/上海主板_历史数据_2018至今_20250729_233546_副本2.csv'
OUTPUT_CSV_PATH = 'V32_AI_Elite_Trades.csv'

# V6核心参数
V_BOTTOM_LOOKBACK = 20
STOP_LOSS_PCT = 0.031
MAX_HOLDING_DAYS = 10
TOP_PERCENTILE = 0.3  # Top 30%信号

# ==============================================================================
# 数据加载与预处理
# ==============================================================================
def load_and_preprocess_data(file_path):
    """加载数据并计算必要的技术指标"""
    print("开始加载数据...")
    start_time = time.time()
    
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载 {len(df):,} 条数据记录")
    except FileNotFoundError:
        print(f"错误：数据文件未找到！")
        return None

    # 标准化列名
    df.rename(columns={
        'ts_code': '代码', 'trade_date': '日期', 'open': '开盘',
        'high': '最高', 'low': '最低', 'close': '收盘',
        'vol': '成交量', 'amount': '成交额'
    }, inplace=True)
    
    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
    df.sort_values(by=['代码', '日期'], inplace=True)
    
    # 计算必要的技术指标
    print("⏳ 计算技术指标...")
    grouped = df.groupby('代码')
    df['ma5'] = grouped['收盘'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f} 秒")
    return df

# ==============================================================================
# AI精英特征提取 (Top 5特征)
# ==============================================================================
def extract_elite_features(stock_data, signal_idx):
    """提取AI发现的Top 5最重要特征"""
    try:
        if signal_idx < 4:  # 需要4天数据
            return None
            
        # 获取4天窗口数据
        window_data = stock_data.iloc[signal_idx-3:signal_idx+1]
        if len(window_data) != 4:
            return None
            
        features = {}
        
        # 获取基础数据
        opens = window_data['开盘'].values
        highs = window_data['最高'].values
        lows = window_data['最低'].values
        closes = window_data['收盘'].values
        volumes = window_data['成交量'].values
        amounts = window_data['成交额'].values if '成交额' in window_data.columns else volumes * closes
        
        current_data = window_data.iloc[-1]
        
        # 1. ma5_trend (最重要特征 15.44%)
        if signal_idx >= 9:  # 需要足够数据计算MA5趋势
            ma5_current = stock_data.iloc[signal_idx-4:signal_idx+1]['收盘'].mean()
            ma5_prev = stock_data.iloc[signal_idx-9:signal_idx-4]['收盘'].mean()
            features['ma5_trend'] = (ma5_current - ma5_prev) / ma5_prev if ma5_prev > 0 else 0
        else:
            features['ma5_trend'] = 0
            
        # 2. close_vs_low_4d (第二重要 10.58%)
        min_low_4d = np.min(lows)
        features['close_vs_low_4d'] = closes[-1] / min_low_4d if min_low_4d > 0 else 1
        
        # 3. closing_strength (第三重要 6.73%)
        # 收盘强度 = (收盘价 - 最低价) / (最高价 - 最低价)
        total_ranges = highs - lows
        closing_strengths = []
        for i in range(len(window_data)):
            if total_ranges[i] > 0:
                strength = (closes[i] - lows[i]) / total_ranges[i]
                closing_strengths.append(strength)
            else:
                closing_strengths.append(0.5)
        features['closing_strength'] = np.mean(closing_strengths)
        
        # 4. amount_volume_ratio (第四重要 6.24%)
        # 平均每股成交额
        avg_amounts = amounts / volumes
        features['amount_volume_ratio'] = np.mean(avg_amounts[volumes > 0]) if np.any(volumes > 0) else 0
        
        # 5. max_intraday_range (第五重要 6.05%)
        # 最大日内波动率
        intraday_ranges = (highs - lows) / closes
        features['max_intraday_range'] = np.max(intraday_ranges)
        
        return features
        
    except Exception as e:
        print(f"特征提取错误: {e}")
        return None

# ==============================================================================
# AI精英评分 (基于Top 5特征)
# ==============================================================================
def calculate_elite_score(features):
    """基于AI发现的重要性权重计算精英评分"""
    if not features:
        return 0
    
    # AI发现的特征重要性权重
    weights = {
        'ma5_trend': 0.154411,
        'close_vs_low_4d': 0.105826,
        'closing_strength': 0.067334,
        'amount_volume_ratio': 0.062353,
        'max_intraday_range': 0.060514
    }
    
    score = 0
    total_weight = 0
    
    for feature, weight in weights.items():
        if feature in features:
            # 标准化特征值到0-1范围
            feature_value = features[feature]
            
            if feature == 'ma5_trend':
                # MA5趋势：正值越大越好
                normalized_value = max(0, min(1, (feature_value + 0.1) / 0.2))
            elif feature == 'close_vs_low_4d':
                # 收盘价相对位置：越接近高点越好
                normalized_value = max(0, min(1, (feature_value - 1) / 0.5))
            elif feature == 'closing_strength':
                # 收盘强度：0-1之间，越大越好
                normalized_value = max(0, min(1, feature_value))
            elif feature == 'amount_volume_ratio':
                # 成交额比率：适中为好，过高过低都不好
                if feature_value > 0:
                    normalized_value = max(0, min(1, np.log(feature_value + 1) / 5))
                else:
                    normalized_value = 0
            elif feature == 'max_intraday_range':
                # 日内波动：适度波动为好
                normalized_value = max(0, min(1, 1 - abs(feature_value - 0.05) / 0.1))
            else:
                normalized_value = 0
                
            score += normalized_value * weight
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0

# ==============================================================================
# V6信号检测与AI评分
# ==============================================================================
def find_ai_elite_signals(all_data):
    """寻找V6信号并进行AI精英评分"""
    print("开始寻找V6信号并进行AI精英评分...")
    start_time = time.time()
    
    all_signals = []
    grouped = all_data.groupby('代码')
    
    for code, stock_data in grouped:
        if len(stock_data) < V_BOTTOM_LOOKBACK + 10:
            continue
            
        df = stock_data.copy().reset_index(drop=True)
        closes = df['收盘'].values
        
        # 寻找V6信号点 (3天版)
        for i in range(V_BOTTOM_LOOKBACK + 4, len(closes) - 2):
            # V6条件1: 连续3天不减
            if not (closes[i-2] <= closes[i-1] <= closes[i]):
                continue
                
            # V6条件2: V型底
            day0_idx = i - 3
            window_start = max(0, day0_idx - V_BOTTOM_LOOKBACK + 1)
            window_closes = closes[window_start:day0_idx + 1]
            if closes[day0_idx] != min(window_closes):
                continue
                
            # 提取AI精英特征
            features = extract_elite_features(df, i)
            if not features:
                continue
                
            # 计算AI精英评分
            ai_score = calculate_elite_score(features)
            
            all_signals.append({
                '代码': code,
                '日期': df.iloc[i]['日期'],
                '买入价格': closes[i],
                'AI精英评分': ai_score
            })
    
    signals_df = pd.DataFrame(all_signals)
    print(f"信号生成完成，共找到 {len(signals_df)} 个信号，耗时: {time.time() - start_time:.2f} 秒")
    return signals_df

# ==============================================================================
# AI精英回测
# ==============================================================================
def run_ai_elite_backtest(all_data, signals_df):
    """执行AI精英策略回测"""
    print(f"开始AI精英回测 (Top {TOP_PERCENTILE*100:.0f}%)...")
    start_time = time.time()
    
    if signals_df.empty:
        return pd.DataFrame()
    
    # 筛选Top 30%精英信号
    threshold = signals_df['AI精英评分'].quantile(1 - TOP_PERCENTILE)
    elite_signals = signals_df[signals_df['AI精英评分'] >= threshold].copy()
    
    print(f"筛选出 {len(elite_signals)} 个精英信号 (评分阈值: {threshold:.4f})")
    
    if elite_signals.empty:
        return pd.DataFrame()
    
    # 执行回测
    trades = []
    active_trades = {}
    
    unique_dates = sorted(elite_signals['日期'].unique())
    stock_groups = all_data.groupby('代码')
    
    for current_date in unique_dates:
        # 清理到期持仓
        ended_stocks = [code for code, end_date in active_trades.items() if current_date >= end_date]
        for code in ended_stocks:
            del active_trades[code]
            
        daily_signals = elite_signals[elite_signals['日期'] == current_date]
        
        for _, signal in daily_signals.iterrows():
            stock_code = signal['代码']
            
            if stock_code in active_trades:
                continue
                
            if stock_code not in stock_groups.groups:
                continue
                
            stock_data = stock_groups.get_group(stock_code)
            future_data = stock_data[stock_data['日期'] > current_date].head(MAX_HOLDING_DAYS + 2)
            
            if len(future_data) < 1:
                continue
                
            # V6交易逻辑 (简化版)
            entry_price = signal['买入价格']
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            
            # 模拟持仓
            remaining_shares = 1.0
            total_pnl = 0.0
            prev_close = entry_price
            is_first_drop = True
            
            sell_date = None
            exit_reason = f"持有{MAX_HOLDING_DAYS}天到期"
            
            for i in range(min(len(future_data), MAX_HOLDING_DAYS)):
                current_row = future_data.iloc[i]
                current_close = current_row['收盘']
                
                # 硬止损检查
                if current_close < stop_loss_price:
                    sell_date = current_row['日期']
                    remaining_pnl = remaining_shares * (stop_loss_price / entry_price - 1)
                    total_pnl += remaining_pnl
                    remaining_shares = 0
                    exit_reason = "硬止损"
                    break
                
                # 分级减仓
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
            
            # 最终退出
            if sell_date is None:
                exit_idx = min(MAX_HOLDING_DAYS - 1, len(future_data) - 1)
                sell_date = future_data.iloc[exit_idx]['日期']
                exit_price = future_data.iloc[exit_idx]['收盘']
                if remaining_shares > 0:
                    remaining_pnl = remaining_shares * (exit_price / entry_price - 1)
                    total_pnl += remaining_pnl
            
            trades.append({
                '代码': stock_code,
                '买入日期': current_date,
                '买入价格': entry_price,
                '卖出日期': sell_date,
                '总收益率': total_pnl,
                '退出原因': exit_reason,
                'AI精英评分': signal['AI精英评分']
            })
            
            active_trades[stock_code] = sell_date
    
    trades_df = pd.DataFrame(trades)
    print(f"AI精英回测完成: {len(trades_df)} 笔交易，耗时: {time.time() - start_time:.2f} 秒")
    return trades_df

# ==============================================================================
# 性能分析
# ==============================================================================
def analyze_ai_elite_performance(trades_df):
    """分析AI精英策略性能"""
    if trades_df.empty:
        print("AI精英回测未产生任何交易")
        return
        
    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['总收益率'] > 0]
    win_rate = len(win_trades) / total_trades
    avg_return = trades_df['总收益率'].mean()
    avg_win = win_trades['总收益率'].mean() if not win_trades.empty else 0
    avg_loss = trades_df[trades_df['总收益率'] <= 0]['总收益率'].mean()
    
    print("\n" + "="*70)
    print("V32 AI精英策略性能分析 (Top 5特征)")
    print("="*70)
    print(f"总交易次数: {total_trades:,}")
    print(f"胜率: {win_rate:.2%}")
    print(f"期望收益: {avg_return:.4%}")
    print(f"平均盈利: {avg_win:.4%}")
    print(f"平均亏损: {avg_loss:.4%}")
    if avg_loss != 0:
        print(f"盈亏比: {-avg_win / avg_loss:.2f}")
    
    print(f"\n--- 与基准对比 ---")
    print(f"V6基准(3天版): 1.89%")
    print(f"V32 AI精英: {avg_return:.4%}")
    
    if avg_return >= 0.02:
        print("🎉 突破2%目标！")
    elif avg_return > 0.0189:
        print("✅ 超越V6基准！")
    else:
        print("❌ 需要进一步优化")
    
    # 评分分布分析
    print(f"\n--- AI评分分布 ---")
    print(f"最高评分: {trades_df['AI精英评分'].max():.4f}")
    print(f"最低评分: {trades_df['AI精英评分'].min():.4f}")
    print(f"平均评分: {trades_df['AI精英评分'].mean():.4f}")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    # 1. 加载数据
    all_data = load_and_preprocess_data(DATA_PATH)
    if all_data is None:
        return
        
    # 2. 寻找AI精英信号
    signals_df = find_ai_elite_signals(all_data)
    
    if signals_df.empty:
        print("未找到任何信号")
        return
        
    # 3. AI精英回测
    trades_df = run_ai_elite_backtest(all_data, signals_df)
    
    # 4. 保存和分析结果
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        analyze_ai_elite_performance(trades_df)
    
    print(f"\n交易记录已保存: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    main()