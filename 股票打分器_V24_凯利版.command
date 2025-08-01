#!/bin/zsh

# 股票打分器_V24_凯利版.command (V24.3 - ATR计算修复版)
# 修复了ATR计算导致的DataFrame赋值错误，现在可以正确处理tushare格式数据

if [ -n "$1" ]; then
    FILE_PATH="$1"
else
    echo "请将CSV数据文件拖拽到此窗口中，然后按 Enter 键:"
    read -r FILE_PATH
fi
FILE_PATH=$(echo "$FILE_PATH" | sed "s/'//g" | sed 's/\\ / /g' | xargs)
if [ ! -f "$FILE_PATH" ]; then
    echo "❌ 错误：文件不存在或路径不正确。"
    read -p "按任意键退出..."
    exit 1
fi
echo "✅ 文件已找到: $(basename "$FILE_PATH")"
echo "🧠 正在启动AI分析引擎 (V24.3 ATR计算修复版)..."
echo "========================================================================"
echo ""

python3 - "$FILE_PATH" <<'EOF'

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 凯利性能数据
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

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rs = rs.fillna(0)
    rs = rs.replace([np.inf, -np.inf], 999)
    return 100 - (100 / (1 + rs))

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    try:
        try: df = pd.read_csv(file_path, low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
        
        # 修复：正确的列名映射
        if 'ts_code' in df.columns:
            df.rename(columns={'ts_code': '股票代码'}, inplace=True)
        if 'trade_date' in df.columns:
            df.rename(columns={'trade_date': '日期'}, inplace=True)
        
        # 其他可能的列名
        rename_map = {'close': '收盘', 'open': '开盘','high': '最高', 'low': '最低', 'vol': '成交量', 'amount': '成交额', 'volume': '成交量'}
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = ['股票代码', '日期', '开盘', '最高', '最低', '收盘', '成交量']
        missing = [col for col in required_cols if col not in df.columns]
        if missing: sys.exit(f"❌ 错误: 缺少列 {missing}")

        # 修复：正确的日期解析
        df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d', errors='coerce')
        df.dropna(subset=['日期', '收盘'], inplace=True)
        
        numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量']
        if '成交额' in df.columns: numeric_cols.append('成交额')
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)

        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        
        print("⏳ 正在计算技术指标...")
        grouped = df.groupby('股票代码', group_keys=False)
        
        df['rsi_14'] = grouped['收盘'].apply(lambda x: calculate_rsi(x, 14))
        df['rsi_30'] = grouped['收盘'].apply(lambda x: calculate_rsi(x, 30))
        
        # 修复ATR计算 - 避免DataFrame赋值错误
        df['high_low_diff'] = df['最高'] - df['最低']
        df['atr'] = grouped['high_low_diff'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df['atr_ratio'] = df['atr'] / df['收盘']
        df.drop('high_low_diff', axis=1, inplace=True)  # 清理临时列
        
        df['ma5'] = grouped['收盘'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['ma10'] = grouped['收盘'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['avg_vol_20'] = grouped['成交量'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        
        if '成交额' in df.columns:
            df['avg_amount_20'] = grouped['成交额'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        else:
            df['avg_amount_20'] = df['avg_vol_20'] * df['收盘']
            
        print("✅ 技术指标计算完成")
        return df
    except Exception as e:
        sys.exit(f"❌ 处理文件错误: {e}")

def extract_features(stock_data, i):
    try:
        row = stock_data.iloc[i]
        prev_row = stock_data.iloc[i-1] if i > 0 else row
        
        open_p, close_p, high_p, low_p = row['开盘'], row['收盘'], row['最高'], row['最低']
        entity = abs(close_p - open_p)
        full_range = high_p - low_p if high_p > low_p else 1e-6
        
        features = {
            'rsi_14': row['rsi_14'], 'rsi_30': row['rsi_30'], 'atr_ratio': row['atr_ratio'],
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
             else: break
        features['consecutive_up_days'] = consecutive_up
        
        # 底部深度
        if i >= 20:
            min_low = stock_data.iloc[max(0, i-20):i]['最低'].min()
            features['bottom_depth'] = (close_p - min_low) / min_low if min_low > 0 else 0
        else:
            features['bottom_depth'] = 0
            
        return features
    except: return None

def find_v6_signals(all_data: pd.DataFrame) -> list:
    print("🔍 正在扫描V6信号 (V24.4 - 精准V底逻辑)...")
    signals = []
    latest_date = all_data['日期'].max()
    print(f"最新交易日: {latest_date.strftime('%Y-%m-%d')}")
    
    # 使用 .loc 提高性能和准确性
    all_data = all_data.set_index('日期')
    
    for stock_code in all_data['股票代码'].unique():
        stock_data = all_data[all_data['股票代码'] == stock_code].sort_index()
        if len(stock_data) < 30: continue
        
        # 只检查最新交易日
        if stock_data.index[-1] != latest_date: continue
        
        # i 是最新日期的行号
        i = len(stock_data) - 1
        
        # 条件1: 4天连涨 (A, B, C, D)
        # D=i, C=i-1, B=i-2, A=i-3
        if i < 3: continue
        closes = stock_data['收盘'].values
        if not (closes[i-3] <= closes[i-2] <= closes[i-1] <= closes[i]): continue
            
        # 条件2: V型底 - '第0天' (A的前一天) 是过去n天的最低点
        day_0_idx = i - 4
        n = 20
        
        # 必须有足够的数据来回溯
        if day_0_idx < n -1: continue

        # V底窗口: 从 '第0天' (day_0_idx) 往前回溯 n 天
        window_start_idx = day_0_idx - n + 1
        
        day_0_close = closes[day_0_idx]
        window_closes = closes[window_start_idx : day_0_idx + 1]
        
        # 判断'第0天'的收盘价是否为窗口最低价
        if day_0_close != np.min(window_closes): continue

        # --- V6信号确认 ---
        features = extract_features(stock_data.reset_index(), i)
        if features:
            features.update({
                '股票代码': stock_code,
                '股票名称': stock_data.iloc[i].get('股票名称', 'N/A'),
                '信号日期': stock_data.index[i],
                '收盘价': stock_data.iloc[i]['收盘']
            })
            signals.append(features)
            
    print(f"✅ 找到 {len(signals)} 个V6信号")
    return signals

def simple_ai_scoring(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty: return signals_df
    print("🧠 正在AI评分...")
    
    feature_weights = {'rsi_30': 0.25, 'bottom_depth': 0.25, 'atr_ratio': 0.15, 'rsi_14': 0.10, 'gap_open': 0.10, 'vol_amp': 0.08, 'lower_shadow_ratio': 0.07}
    scores = np.zeros(len(signals_df))
    
    for feature, weight in feature_weights.items():
        if feature in signals_df.columns:
            values = signals_df[feature].fillna(0)
            if values.std() > 0:
                normalized = (values - values.min()) / (values.max() - values.min())
                scores += normalized * weight
    
    signals_df['AI评分'] = scores
    return signals_df.sort_values('AI评分', ascending=False)

def assign_kelly_position(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty: return signals_df
    
    signals_df['排名百分位'] = signals_df['AI评分'].rank(pct=True) * 100
    signals_df['性能档位'] = np.floor(signals_df['排名百分位']).astype(int)
    
    eligible = signals_df[signals_df['性能档位'].isin(KELLY_PERFORMANCE_LOOKUP.keys())].copy()
    if eligible.empty: return pd.DataFrame()
    
    eligible['W'] = eligible['性能档位'].map(lambda x: KELLY_PERFORMANCE_LOOKUP[x]['W'])
    eligible['R'] = eligible['性能档位'].map(lambda x: KELLY_PERFORMANCE_LOOKUP[x]['R'])
    
    W, R = eligible['W'], eligible['R']
    kelly_pct = np.maximum(0, (W - (1 - W) / R) * 100)
    
    eligible['激进仓位'] = kelly_pct
    eligible['中立仓位'] = kelly_pct / 2
    eligible['保守仓位'] = kelly_pct / 4
    
    return eligible.sort_values('排名百分位', ascending=False)

def main():
    if len(sys.argv) < 2: sys.exit("❌ 需要数据文件")
    file_path = sys.argv[1]
    
    all_data = load_and_prepare_data(file_path)
    if all_data.empty: sys.exit("❌ 数据为空")
    
    print(f"ℹ️  数据文件最新交易日: {all_data['日期'].max().strftime('%Y-%m-%d')}")
    print(f"📊 数据包含 {all_data['股票代码'].nunique()} 只股票")
    
    signals = find_v6_signals(all_data)
    if not signals: 
        print("\n✅ 分析完成：没有股票满足V6信号条件")
        return
    
    scored_signals = simple_ai_scoring(pd.DataFrame(signals))
    final_signals = assign_kelly_position(scored_signals)
    
    print("\n" + "="*80)
    print("📊 V24.3 凯利版决策报告")
    print("="*80)
    
    if final_signals.empty:
        print("所有信号均不在有效收益区间")
    else:
        print(f"筛选出 {len(final_signals)} 个优质信号:")
        
        display_cols = ['股票代码', '股票名称', '收盘价', 'AI评分', '排名百分位', '激进仓位', '中立仓位', '保守仓位']
        display_df = final_signals[display_cols].copy()
        display_df['排名'] = range(1, len(display_df) + 1)
        display_df = display_df[['排名'] + display_cols]

        for col in ['激进仓位', '中立仓位', '保守仓位', '排名百分位']:
             display_df[col] = display_df[col].map('{:.1f}%'.format)
        display_df['AI评分'] = display_df['AI评分'].map('{:.3f}'.format)
        
        print(display_df.to_string(index=False))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"V24_凯利版决策_{timestamp}.csv"
        final_signals.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果已保存: {output_file}")
    
    print("\n" + "="*80)
    print("✅ V24.3 分析完成")

if __name__ == '__main__':
    main()

EOF

echo ""
echo "✅ V24.3 凯利版打分器 ATR计算修复版 运行完毕！"
echo "按 Enter 键退出..."
read
exit 0