#!/bin/bash
# A股潜力股票打分器 (V6.2 - 当日机会扫描版)

clear
echo "🚀 A股潜力股票打分器 (V6.2 - 当日机会扫描版) 🚀"
echo "========================================================================"
echo "本工具基于'V6.1-反传统'回测策略，并只扫描数据文件中的【最新交易日】。"
echo "核心规则: 均线排列 + 上影线限制 + 成交量不放大"
echo "========================================================================"
echo ""

# 修改为优先使用命令行参数，如果未提供则回退到交互模式
if [ -n "$1" ]; then
    FILE_PATH="$1"
else
echo "请将 '连续上涨股票详细数据_*.csv' 文件拖拽到此窗口中，然后按 Enter 键:"
read -r FILE_PATH
fi

# 清理文件路径（macOS拖拽可能会添加不必要的引号和转义符）
FILE_PATH=$(echo "$FILE_PATH" | sed "s/'//g" | sed 's/\\ / /g' | xargs)

# 检查文件是否存在
if [ ! -f "$FILE_PATH" ]; then
    echo "❌ 错误：文件不存在或路径不正确。"
    echo "路径: '$FILE_PATH'"
    echo ""
    read -p "按任意键退出..."
    exit 1
fi

echo "✅ 文件已找到: $(basename "$FILE_PATH")"
echo "🐍 正在启动Python分析引擎..."
echo "========================================================================"
echo ""

# 使用heredoc将Python脚本嵌入到bash脚本中
python3 - "$FILE_PATH" <<'EOF'

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# --- V6.4 "动态窗口打分版" 核心参数 ---
RISE_DAYS = 4
V_BOTTOM_WINDOW = 20  # 基础筛选窗口
VOL_RATIO_MAX = 1.8
UPPER_SHADOW_TO_ENTITY_MAX = 0.5

# 历史回测数据 (20天窗口v2策略)
HIST_WIN_RATE = 0.6211
HIST_PROFIT_LOSS_RATIO = 3.30


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """加载、合并、清洗并预计算所有需要的数据"""
    try:
        # 增加对GBK编码的兼容
        try:
            df = pd.read_csv(file_path, dtype={'trade_date': str})
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, dtype={'trade_date': str}, encoding='gbk')

        # --- 列名兼容性处理 ---
        rename_map = {
            'date': 'trade_date', 'vol': 'volume', 'stock_name': 'name', 
            '股票名称': 'name', '代码': 'ts_code', '名称': 'name'
        }
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"❌ 错误: 输入文件缺少以下必要列: {', '.join(missing_cols)}", file=sys.stderr)
            sys.exit(1)

        # --- 数据清洗和排序 ---
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df.dropna(subset=['trade_date'], inplace=True)
        df.sort_values(['ts_code', 'trade_date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close'], inplace=True)

        # --- 预计算指标 (向量化操作，提升性能) ---
        df_grouped = df.groupby('ts_code')

        # 为核心规则预计算列
        df['entity'] = np.abs(df['close'] - df['open'])
        df['ma5'] = df_grouped['close'].transform(lambda x: x.rolling(window=5).mean())
        df['ma10'] = df_grouped['close'].transform(lambda x: x.rolling(window=10).mean())
        df['ma20'] = df_grouped['close'].transform(lambda x: x.rolling(window=20).mean())
        df['prev_vol'] = df_grouped['volume'].shift(1)
        df['rolling_min_v_window'] = df_grouped['close'].transform(lambda x: x.rolling(window=V_BOTTOM_WINDOW, min_periods=V_BOTTOM_WINDOW).min())
        
        return df
    
    except Exception as e:
        print(f"读取或处理CSV文件时出错: {e}")
        sys.exit(1)

def find_base_candidates(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    寻找所有满足“4日连涨 + V型底”基础形态的股票。
    这是一个向量化的筛选过程，以提高效率。
    """
    df = all_data.copy()
    
    # 核心入场形态 (向量化)
    # 1. 4连涨
    c1 = df['close'] > df.groupby('ts_code')['close'].shift(1)
    c2 = df.groupby('ts_code')['close'].shift(1) > df.groupby('ts_code')['close'].shift(2)
    c3 = df.groupby('ts_code')['close'].shift(2) > df.groupby('ts_code')['close'].shift(3)
    c4 = df.groupby('ts_code')['close'].shift(3) > df.groupby('ts_code')['close'].shift(4)
    is_4_day_rise = c1 & c2 & c3 & c4

    # 2. V型底确认 (基础20天)
    anchor_close = df.groupby('ts_code')['close'].shift(RISE_DAYS)
    rolling_min_at_anchor = df.groupby('ts_code')['rolling_min_v_window'].shift(RISE_DAYS)
    is_v_bottom = (anchor_close == rolling_min_at_anchor)
    
    base_signal_mask = is_4_day_rise & is_v_bottom
    
    candidate_indices = df[base_signal_mask].index
    candidates_df = df.loc[candidate_indices]
    
    # 计算实际v_depth
    def calc_v_depth(ts_code, global_idx):
        stock_data = df[df['ts_code'] == ts_code].iloc[:global_idx + 1].reset_index(drop=True)
        local_idx = len(stock_data) - 1  # 信号日本地索引
        anchor_local = local_idx - RISE_DAYS
        if anchor_local < 0:
            return 0
        min_price = stock_data['close'].iloc[anchor_local]
        for days_back in range(1, anchor_local + 1):  # 从锚定日前一天回看
            if stock_data['close'].iloc[anchor_local - days_back] < min_price:
                return days_back
        return anchor_local  # 如果无更低点，返回到数据开始的深度
    
    candidates_df['v_depth'] = [calc_v_depth(row['ts_code'], i) for i, row in candidates_df.iterrows()]
    
    latest_candidates = candidates_df.loc[candidates_df.groupby('ts_code')['trade_date'].idxmax()]
    
    return latest_candidates

def apply_core_filters(candidate_row: pd.Series) -> list[str]:
    """
    对单个候选股应用V6.1的三条核心筛选规则。
    返回一个包含所有失败原因的列表。如果列表为空，则表示全部通过。
    """
    failures = []
    
    # 规则1: 均线多头排列
    if not (candidate_row['ma5'] > candidate_row['ma10'] > candidate_row['ma20']):
        failures.append("均线非多头")
    
    # 规则2: 上影线限制
    upper_shadow = candidate_row['high'] - max(candidate_row['open'], candidate_row['close'])
    entity = candidate_row['entity']
    if entity > 0 and (upper_shadow / entity) >= UPPER_SHADOW_TO_ENTITY_MAX:
        failures.append(f"上影线过长(>{UPPER_SHADOW_TO_ENTITY_MAX:.0%})")
        
    # 规则3: 成交量不放大
    if candidate_row['volume'] > VOL_RATIO_MAX * candidate_row['prev_vol']:
        failures.append(f"成交量放大(>{VOL_RATIO_MAX}倍)")

    return failures

# 新函数: 计算仓位建议
def calculate_position(failures: list, v_depth: int) -> dict:
    # 基线凯利
    kelly_f = (HIST_WIN_RATE * (HIST_PROFIT_LOSS_RATIO + 1) - 1) / HIST_PROFIT_LOSS_RATIO
    
    aggressive = kelly_f / 2
    conservative = kelly_f / 4
    
    # 基于v_depth调整 (越长越好，最大+50%)
    adjustment_depth = 1 + min(max((v_depth - 20) / 60, 0), 1) * 0.5
    
    num_failures = len(failures)
    if num_failures == 0:
        adjustment_fail = 1.0
    elif num_failures == 1:
        adjustment_fail = 0.75
    else:
        adjustment_fail = 0.5
        
    final_adjust = adjustment_depth * adjustment_fail
    
    return {
        'aggressive': aggressive * final_adjust * 100,
        'conservative': conservative * final_adjust * 100
    }


def main():
    if len(sys.argv) < 2:
        print("❌ 错误: 请提供数据文件路径作为参数。")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    all_data = load_and_prepare_data(file_path)
    
    if all_data.empty:
        print("🤷 数据加载后为空，无法继续分析。")
        return
        
    latest_date_in_file = all_data['trade_date'].max()
    print(f"ℹ️  数据文件最新交易日为: {latest_date_in_file.strftime('%Y-%m-%d')}")
        
    print(f"⏳ 正在从 {all_data['ts_code'].nunique()} 只股票中寻找基础候选股...")
    
    # 1. 寻找所有历史上满足“4连涨+V型底”的候选
    candidates = find_base_candidates(all_data)
    
    if candidates.empty:
        print("\n✅ 分析完成：在所有股票中，没有找到满足“4日连涨+20日V型底”基础形态的股票。")
        return

    # 关键步骤：只筛选出信号日期为最新交易日的股票
    final_candidates = candidates[candidates['trade_date'] == latest_date_in_file].copy()

    print(f"✅ 在最新交易日找到 {len(final_candidates)} 个基础候选股，正在进行核心规则筛选...")
    
    if final_candidates.empty:
        print("\n✅ 分析完成：在最新交易日，没有股票满足基础形态。")
        return

    results = []
    for _, row in final_candidates.iterrows():
        failures = apply_core_filters(row)
        position = calculate_position(failures, int(row['v_depth']))
        results.append({
            'ts_code': row['ts_code'],
            'name': row.get('name', 'N/A'),
            'trade_date': row['trade_date'].strftime('%Y-%m-%d'),
            'failures': failures,
            'status': '✅ 理想' if not failures else '⚠️ 警告',
            'v_depth': row['v_depth'],
            'aggressive_pos': f"{position['aggressive']:.1f}%",
            'conservative_pos': f"{position['conservative']:.1f}%"
        })
        
    # --- 生成最终报告 ---
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['status', 'ts_code'], ascending=[True, True], inplace=True)
    
    ideal_stocks = results_df[results_df['status'] == '✅ 理想']
    warning_stocks = results_df[results_df['status'] == '⚠️ 警告']
    
    print("\n" + "="*80)
    print("📊 最终筛选报告")
    print("="*80)
    
    if not ideal_stocks.empty:
        print(f"\n--- ✅ 理想信号 ({len(ideal_stocks)}只) ---")
        print("全部通过三项核心避险规则，是最高质量的信号。")
        ideal_stocks = ideal_stocks.rename(columns={'v_depth': 'V底深度(天)'})
        print(ideal_stocks[['ts_code', 'name', 'trade_date', 'V底深度(天)', 'aggressive_pos', 'conservative_pos']].to_string(index=False))
    else:
        print("\n--- ✅ 理想信号 (0只) ---")
        print("没有股票能完美通过所有核心规则。")

    if not warning_stocks.empty:
        print(f"\n--- ⚠️ 警告信号 ({len(warning_stocks)}只) ---")
        print("满足基础形态，但未通过部分核心规则，请注意风险。")
        warning_stocks['details'] = warning_stocks['failures'].apply(lambda x: ', '.join(x))
        warning_stocks = warning_stocks.rename(columns={'v_depth': 'V底深度(天)'})
        print(warning_stocks[['ts_code', 'name', 'trade_date', 'details', 'V底深度(天)', 'aggressive_pos', 'conservative_pos']].to_string(index=False))
    
    print("\n" + "="*80)
    print("分析完成。")
    print("仓位调整: 基于实际V底深度 (越长越好，最大+50%) 和失败规则。")
    print("V底深度信息: 显示实际从锚定日回看的最低点天数 (越长表示更强的底部支撑)。")


if __name__ == '__main__':
    main()

EOF
chmod +x "$0"
echo ""
echo "✅ 打分器脚本已成功更新至 V6.5 - 增强信息版。"
read -p "按 Enter 键退出..."
exit 0 