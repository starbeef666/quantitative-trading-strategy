import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (V24凯利版策略)
# ==============================================================================
DATA_PATH = '上海主板_历史数据_2018至今_20250729_233546_副本2.csv'
OUTPUT_CSV_PATH = 'V27_V24_Kelly_Trades.csv'

# V6 核心参数
V_BOTTOM_LOOKBACK = 20  # V型底回看天数 (n≥20)
STOP_LOSS_PCT = 0.031   # 固定止损比例
MAX_HOLDING_DAYS = 10   # 最大持仓天数

# V24凯利性能数据表 (来自打分器)
KELLY_PERFORMANCE_LOOKUP = {
    0: {'W': 0.700, 'R': 5.66}, 1: {'W': 0.667, 'R': 3.94}, 2: {'W': 0.621, 'R': 3.66},
    3: {'W': 0.591, 'R': 2.78}, 4: {'W': 0.520, 'R': 2.74}, 5: {'W': 0.502, 'R': 2.63},
    6: {'W': 0.467, 'R': 2.43}, 7: {'W': 0.482, 'R': 2.66}, 8: {'W': 0.464, 'R': 2.74},
    9: {'W': 0.430, 'R': 2.57}, 10: {'W': 0.433, 'R': 2.62}, 11: {'W': 0.437, 'R': 2.69},
    12: {'W': 0.416, 'R': 3.18}, 13: {'W': 0.430, 'R': 2.76}, 14: {'W': 0.394, 'R': 2.55},
    15: {'W': 0.400, 'R': 2.68}, 16: {'W': 0.423, 'R': 2.70}, 17: {'W': 0.417, 'R': 2.77},
    18: {'W': 0.439, 'R': 3.18}, 19: {'W': 0.405, 'R': 2.54}, 20: {'W': 0.456, 'R': 2.78},
    21: {'W': 0.418, 'R': 2.83}, 22: {'W': 0.439, 'R': 2.71}, 23: {'W': 0.402, 'R': 2.75},
    24: {'W': 0.437, 'R': 2.55}, 25: {'W': 0.459, 'R': 2.67}, 26: {'W': 0.408, 'R': 2.76},
    27: {'W': 0.452, 'R': 2.43}, 28: {'W': 0.427, 'R': 2.78}, 29: {'W': 0.426, 'R': 2.53},
    30: {'W': 0.425, 'R': 2.82}, 31: {'W': 0.419, 'R': 2.54}, 32: {'W': 0.421, 'R': 2.78},
    33: {'W': 0.406, 'R': 2.35}, 35: {'W': 0.375, 'R': 2.63}, 36: {'W': 0.385, 'R': 2.40},
    37: {'W': 0.409, 'R': 2.66}, 38: {'W': 0.432, 'R': 2.23}, 39: {'W': 0.367, 'R': 2.63}
}

# ==============================================================================
# 数据加载与预处理
# ==============================================================================
def calculate_rsi(series, period):
    """计算RSI指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rs = rs.fillna(0)
    rs = rs.replace([np.inf, -np.inf], 999)
    return 100 - (100 / (1 + rs))

def load_and_preprocess_data(file_path):
    """加载数据并计算所有技术指标"""
    print(f"开始加载数据...")
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
    
    print("⏳ 正在计算技术指标...")
    grouped = df.groupby('代码')
    
    # 计算所有V24需要的技术指标
    df['rsi_14'] = grouped['收盘'].transform(lambda x: calculate_rsi(x, 14))
    df['rsi_30'] = grouped['收盘'].transform(lambda x: calculate_rsi(x, 30))
    
    # ATR计算 (修复版)
    df['high_low_diff'] = df['最高'] - df['最低']
    df['atr'] = grouped['high_low_diff'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df['atr_ratio'] = df['atr'] / df['收盘']
    df.drop('high_low_diff', axis=1, inplace=True)
    
    # 移动平均线
    df['ma5'] = grouped['收盘'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['ma10'] = grouped['收盘'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['avg_vol_20'] = grouped['成交量'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # 成交额处理
    if '成交额' in df.columns:
        df['avg_amount_20'] = grouped['成交额'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    else:
        df['avg_amount_20'] = df['avg_vol_20'] * df['收盘']
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f} 秒")
    return df

# ==============================================================================
# V6核心信号检测
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
                
            # 记录信号和完整的股票数据
            signals.append({
                '日期': df.iloc[i]['日期'],
                '代码': code,
                '买入价格': closes[i],
                '数据索引': i,
                '股票数据': df
            })
            
    print(f"V6信号寻找完成，共找到 {len(signals)} 个，耗时: {time.time() - start_time:.2f} 秒")
    return signals

# ==============================================================================
# V24特征提取
# ==============================================================================
def extract_v24_features(stock_data, i):
    """提取V24凯利打分器使用的特征"""
    try:
        row = stock_data.iloc[i]
        prev_row = stock_data.iloc[i-1] if i > 0 else row
        
        open_p, close_p, high_p, low_p = row['开盘'], row['收盘'], row['最高'], row['最低']
        entity = abs(close_p - open_p)
        full_range = high_p - low_p if high_p > low_p else 1e-6
        
        features = {
            'rsi_14': row['rsi_14'],
            'rsi_30': row['rsi_30'],
            'atr_ratio': row['atr_ratio'],
            'body_ratio': entity / full_range,
            'upper_shadow_ratio': (high_p - max(open_p, close_p)) / (entity if entity > 0 else 1e-6),
            'lower_shadow_ratio': (min(open_p, close_p) - low_p) / (entity if entity > 0 else 1e-6),
            'vol_amp': row['成交量'] / row['avg_vol_20'] if row['avg_vol_20'] > 0 else 1,
            'amount_amp': row['成交额'] / row['avg_amount_20'] if row['avg_amount_20'] > 0 else 1,
            'ma5_above_ma10': 1 if row['ma5'] > row['ma10'] else 0,
            'close_above_ma5': 1 if close_p > row['ma5'] else 0,
            'gap_open': (open_p - prev_row['收盘']) / prev_row['收盘'] if i > 0 and prev_row['收盘'] > 0 else 0
        }
        
        # 连续上涨天数
        consecutive_up = 0
        for k in range(i, max(i-10, 0), -1):
            if k > 0 and stock_data.iloc[k]['收盘'] >= stock_data.iloc[k-1]['收盘']:
                consecutive_up += 1
            else:
                break
        features['consecutive_up_days'] = consecutive_up
        
        # 底部深度
        if i >= 20:
            min_low = stock_data.iloc[max(0, i-20):i]['最低'].min()
            features['bottom_depth'] = (close_p - min_low) / min_low if min_low > 0 else 0
        else:
            features['bottom_depth'] = 0
            
        return features
    except:
        return None

# ==============================================================================
# V24 AI评分与凯利筛选
# ==============================================================================
def v24_ai_scoring(signals_df):
    """V24 AI评分算法"""
    if signals_df.empty:
        return signals_df
    
    print("🧠 正在进行V24 AI评分...")
    
    # V24特征权重 (来自打分器代码)
    feature_weights = {
        'rsi_30': 0.25,
        'bottom_depth': 0.25,
        'atr_ratio': 0.15,
        'rsi_14': 0.10,
        'gap_open': 0.10,
        'vol_amp': 0.08,
        'lower_shadow_ratio': 0.07
    }
    
    scores = np.zeros(len(signals_df))
    
    for feature, weight in feature_weights.items():
        if feature in signals_df.columns:
            values = signals_df[feature].fillna(0)
            if values.std() > 0:
                normalized = (values - values.min()) / (values.max() - values.min())
                scores += normalized * weight
    
    signals_df['AI评分'] = scores
    return signals_df.sort_values('AI评分', ascending=False)

def assign_kelly_position(signals_df):
    """V24凯利仓位分配"""
    if signals_df.empty:
        return signals_df
    
    print("📊 正在进行凯利仓位分配...")
    
    # 计算排名百分位
    signals_df['排名百分位'] = signals_df['AI评分'].rank(pct=True) * 100
    signals_df['性能档位'] = np.floor(signals_df['排名百分位']).astype(int)
    
    # 只保留有效档位的信号
    eligible = signals_df[signals_df['性能档位'].isin(KELLY_PERFORMANCE_LOOKUP.keys())].copy()
    
    if eligible.empty:
        print("警告：没有信号落在有效的凯利档位中")
        return pd.DataFrame()
    
    # 映射胜率和盈亏比
    eligible['W'] = eligible['性能档位'].map(lambda x: KELLY_PERFORMANCE_LOOKUP[x]['W'])
    eligible['R'] = eligible['性能档位'].map(lambda x: KELLY_PERFORMANCE_LOOKUP[x]['R'])
    
    # 计算凯利仓位
    W, R = eligible['W'], eligible['R']
    kelly_pct = np.maximum(0, (W - (1 - W) / R) * 100)
    
    eligible['激进仓位'] = kelly_pct
    eligible['中立仓位'] = kelly_pct / 2
    eligible['保守仓位'] = kelly_pct / 4
    
    print(f"凯利筛选完成：从 {len(signals_df)} 个信号中筛选出 {len(eligible)} 个有效信号")
    
    return eligible.sort_values('排名百分位', ascending=False)

# ==============================================================================
# V24凯利回测逻辑
# ==============================================================================
def run_v24_kelly_backtest(all_data, kelly_signals, position_type='中立仓位'):
    """执行V24凯利策略回测"""
    print(f"开始执行V24凯利策略回测 (使用{position_type})...")
    start_time = time.time()
    
    trades = []
    active_trades = {}
    
    if kelly_signals.empty:
        return pd.DataFrame()
        
    unique_dates = sorted(kelly_signals['日期'].unique())
    stock_groups = all_data.groupby('代码')

    for current_date in unique_dates:
        # 清理已到期的持仓
        ended_stocks = [code for code, end_date in active_trades.items() if current_date >= end_date]
        for code in ended_stocks:
            del active_trades[code]
        
        daily_signals = kelly_signals[kelly_signals['日期'] == current_date]
        
        for _, signal in daily_signals.iterrows():
            stock_code = signal['代码']
            
            if stock_code in active_trades:
                continue
            
            entry_price = signal['买入价格']
            buy_date = signal['日期']
            position_size = signal[position_type] / 100  # 转换为小数
            
            # 获取该股票的未来数据
            if stock_code not in stock_groups.groups:
                continue
                
            stock_data = stock_groups.get_group(stock_code)
            future_data = stock_data[stock_data['日期'] > buy_date].head(MAX_HOLDING_DAYS + 2)
            
            if len(future_data) < 1:
                continue

            # V6交易管理 + V24凯利仓位
            remaining_shares = position_size
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
                '排名百分位': signal['排名百分位'],
                '凯利仓位': signal[position_type],
                '胜率预期': signal['W'],
                '盈亏比预期': signal['R']
            })
            
            active_trades[stock_code] = sell_date

    print(f"V24凯利回测完成，耗时: {time.time() - start_time:.2f} 秒")
    return pd.DataFrame(trades)

# ==============================================================================
# 性能分析
# ==============================================================================
def analyze_v24_performance(trades_df, position_type='中立仓位'):
    if trades_df.empty:
        print("V24凯利回测未产生任何交易。")
        return

    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['总收益率'] > 0]
    loss_trades = trades_df[trades_df['总收益率'] <= 0]
    
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    avg_return = trades_df['总收益率'].mean()
    avg_win = win_trades['总收益率'].mean() if not win_trades.empty else 0
    avg_loss = loss_trades['总收益率'].mean() if not loss_trades.empty else 0
    
    # 凯利预期 vs 实际
    expected_win_rate = trades_df['胜率预期'].mean()
    expected_profit_loss_ratio = trades_df['盈亏比预期'].mean()
    actual_profit_loss_ratio = -avg_win / avg_loss if avg_loss != 0 else 0
    
    print("\n" + "="*70)
    print(f"V24.0 凯利策略性能分析 ({position_type})")
    print("="*70)
    print(f"总交易次数: {total_trades:,.0f}")
    print(f"胜率: {win_rate:.2%}")
    print(f"期望收益 (平均总收益率): {avg_return:.4%}")
    print(f"平均盈利收益率: {avg_win:.4%}")
    print(f"平均亏损收益率: {avg_loss:.4%}")
    print(f"实际盈亏比: {actual_profit_loss_ratio:.2f}")
    
    print(f"\n--- 凯利预期 vs 实际对比 ---")
    print(f"凯利预期胜率: {expected_win_rate:.2%}")
    print(f"实际胜率: {win_rate:.2%}")
    print(f"凯利预期盈亏比: {expected_profit_loss_ratio:.2f}")
    print(f"实际盈亏比: {actual_profit_loss_ratio:.2f}")
    
    # 与基准对比
    print(f"\n--- 与基准对比 ---")
    print(f"V6基准期望收益: 1.57%")
    print(f"V24凯利期望收益: {avg_return:.4%}")
    
    if avg_return >= 0.02:
        print("✅ 期望收益达到2%目标！")
    elif avg_return > 0.0157:
        print("✅ 期望收益超越V6基准！")
    else:
        print("❌ 期望收益未超越V6基准")
    
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
        print("未找到任何V6信号，无法进行V24筛选。")
        return
    
    # 3. 提取V24特征
    print("开始提取V24特征...")
    featured_signals = []
    
    for signal in v6_signals:
        features = extract_v24_features(signal['股票数据'], signal['数据索引'])
        if features:
            features.update({
                '日期': signal['日期'],
                '代码': signal['代码'],
                '买入价格': signal['买入价格']
            })
            featured_signals.append(features)
    
    signals_df = pd.DataFrame(featured_signals)
    print(f"成功为 {len(signals_df)} 个信号提取特征")
    
    if signals_df.empty:
        print("特征提取失败，无法进行V24筛选。")
        return
    
    # 4. V24 AI评分
    scored_signals = v24_ai_scoring(signals_df)
    
    # 5. 凯利筛选和仓位分配
    kelly_signals = assign_kelly_position(scored_signals)
    
    if kelly_signals.empty:
        print("凯利筛选后无有效信号，无法进行回测。")
        return
    
    # 6. 执行V24凯利回测 (使用中立仓位)
    trades_df = run_v24_kelly_backtest(all_data, kelly_signals, '中立仓位')
    
    # 7. 保存和分析结果
    if not trades_df.empty:
        trades_df.sort_values(by='买入日期', inplace=True)
        trades_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        analyze_v24_performance(trades_df, '中立仓位')

if __name__ == '__main__':
    main()