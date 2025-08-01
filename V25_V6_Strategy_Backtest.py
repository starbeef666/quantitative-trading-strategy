import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (V6.0 策略 - 无过滤纯净版)
# ==============================================================================
DATA_PATH = '上海主板_历史数据_2018至今_20250729_233546_副本2.csv'
OUTPUT_CSV_PATH = 'V25_V6_Strategy_NoFilter_Trades.csv'

# V6 核心参数
V_BOTTOM_LOOKBACK = 20  # V型底回看天数 (n≥20)
STOP_LOSS_PCT = 0.031   # 固定止损比例
MAX_HOLDING_DAYS = 10   # 最大持仓天数

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
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f} 秒")
    return df

# ==============================================================================
# V6.0 核心信号逻辑 (无过滤纯净版)
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
        
        # 手动循环检查每个可能的信号点
        for i in range(V_BOTTOM_LOOKBACK + 4, len(closes)):
            # 条件1: 连续4天收盘价不减 A≤B≤C≤D
            # i-3=A, i-2=B, i-1=C, i=D
            if not (closes[i-3] <= closes[i-2] <= closes[i-1] <= closes[i]):
                continue
                
            # 条件2: V型底 - 第0天(i-4)是过去n天最低点
            day0_idx = i - 4
            window_start = max(0, day0_idx - V_BOTTOM_LOOKBACK + 1)
            window_closes = closes[window_start:day0_idx + 1]
            
            if closes[day0_idx] != min(window_closes):
                continue
                
            # 所有条件满足，记录信号
            signals.append({
                '日期': df.iloc[i]['日期'],
                '代码': code,
                '买入价格': closes[i]  # 第4天收盘价买入
            })
            
    print(f"信号寻找完成，耗时: {time.time() - start_time:.2f} 秒")
    return signals

# ==============================================================================
# V6.0 核心回测逻辑 (含分级减仓)
# ==============================================================================
def run_v6_backtest(all_data, signals):
    """执行V6.0策略回测"""
    print("开始执行V6.0核心回测...")
    start_time = time.time()
    
    trades = []
    active_trades = {}
    
    if not signals:
        return pd.DataFrame()
        
    signals_df = pd.DataFrame(signals).sort_values(by='日期')
    unique_dates = sorted(signals_df['日期'].unique())
    
    # 为了提高性能，预先按股票分组
    stock_groups = all_data.groupby('代码')

    for current_date in unique_dates:
        # 清理已到期的持仓
        ended_stocks = [code for code, end_date in active_trades.items() if current_date >= end_date]
        for code in ended_stocks:
            del active_trades[code]
        
        daily_signals = signals_df[signals_df['日期'] == current_date]
        
        for _, signal in daily_signals.iterrows():
            stock_code = signal['代码']
            
            if stock_code in active_trades:
                continue
            
            entry_price = signal['买入价格']
            buy_date = signal['日期']
            
            # 获取该股票的未来数据
            if stock_code not in stock_groups.groups:
                continue
                
            stock_data = stock_groups.get_group(stock_code)
            future_data = stock_data[stock_data['日期'] > buy_date].head(MAX_HOLDING_DAYS + 2)
            
            if len(future_data) < 1:
                continue

            # V6 交易管理核心逻辑
            remaining_shares = 1.0
            total_pnl = 0.0
            prev_close = entry_price
            is_first_drop = True
            
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            
            sell_date = None
            exit_reason = f"持有{MAX_HOLDING_DAYS}天到期"

            for i in range(min(len(future_data), MAX_HOLDING_DAYS)):
                current_row = future_data.iloc[i]
                current_close = current_row['收盘']
                
                # 1. 硬止损检查 (最高优先级)
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
                '退出原因': exit_reason
            })
            
            active_trades[stock_code] = sell_date

    print(f"回测完成，耗时: {time.time() - start_time:.2f} 秒")
    return pd.DataFrame(trades)

# ==============================================================================
# 性能分析
# ==============================================================================
def analyze_performance(trades_df):
    if trades_df.empty:
        print("回测未产生任何交易。")
        return

    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['总收益率'] > 0]
    loss_trades = trades_df[trades_df['总收益率'] <= 0]
    
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    avg_return = trades_df['总收益率'].mean()
    avg_win = win_trades['总收益率'].mean() if not win_trades.empty else 0
    avg_loss = loss_trades['总收益率'].mean() if not loss_trades.empty else 0
    
    print("\n" + "="*60)
    print("V6.0 策略性能分析 (无过滤纯净版)")
    print("="*60)
    print(f"总交易次数: {total_trades:,.0f}")
    print(f"胜率: {win_rate:.2%}")
    print(f"期望收益 (平均总收益率): {avg_return:.4%}")
    print(f"平均盈利收益率: {avg_win:.4%}")
    print(f"平均亏损收益率: {avg_loss:.4%}")
    if avg_loss != 0:
        print(f"盈亏比: {-avg_win / avg_loss:.2f}")
    
    # 检查是否达到2%目标
    if avg_return >= 0.02:
        print("✅ 期望收益达到2%目标！")
    else:
        print("❌ 期望收益未达到2%目标")
    
    print(f"\n交易记录已保存至: '{OUTPUT_CSV_PATH}'")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    all_data = load_and_preprocess_data(DATA_PATH)
    if all_data is None:
        return
        
    base_signals = find_v6_base_signals(all_data)
    
    print(f"\n>>> 找到 {len(base_signals)} 个符合V6.0核心入场标准的交易信号。 <<<\n")
    
    if not base_signals:
        print("未找到任何信号，无法进行回测。")
        return
        
    trades_df = run_v6_backtest(all_data, base_signals)
    
    if not trades_df.empty:
        trades_df.sort_values(by='买入日期', inplace=True)
        trades_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        analyze_performance(trades_df)

if __name__ == '__main__':
    main()