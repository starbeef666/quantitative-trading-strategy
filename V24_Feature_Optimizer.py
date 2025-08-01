
import pandas as pd
import numpy as np
import sys
import os
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- V24 特征优化器 ---
# 目标: 自动寻找最优的特征权重组合, 以最大化Top 20%信号的历史平均收益

# --- 核心配置 ---
CACHE_FILE_PATH = 'v23_signals_cache_上海主板_历史数据_2018至今_20250729_233546.csv.pkl'
TOP_PERCENTILE = 0.2
FEATURES = [
    'rsi_14', 'rsi_30', 'atr_ratio', 'body_ratio', 'upper_shadow_ratio',
    'lower_shadow_ratio', 'vol_amp', 'amount_amp', 'ma5_above_ma10',
    'close_above_ma5', 'gap_open', 'consecutive_up_days', 'bottom_depth'
]

# --- 目标函数 ---
def objective_function(weights, signals_df):
    """
    优化器要优化的目标函数。
    输入: 一组权重 (weights) 和包含所有信号的DataFrame。
    输出: 负的平均收益率 (因为优化器默认是寻找最小值)。
    """
    # 1. 计算AI评分
    # 将权重与特征相乘并求和
    scores = np.zeros(len(signals_df))
    for i, feature in enumerate(FEATURES):
        # 简单归一化处理 (0-1之间)
        feature_values = signals_df[feature].fillna(0)
        normalized_feature = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-6)
        scores += weights[i] * normalized_feature
        
    signals_df['AI评分'] = scores
    
    # 2. 筛选Top 20%的信号
    threshold = signals_df['AI评分'].quantile(1 - TOP_PERCENTILE)
    top_signals = signals_df[signals_df['AI评分'] >= threshold]
    
    if top_signals.empty:
        return 0 # 如果没有信号，则收益为0
        
    # 3. 计算这部分信号的平均真实收益
    mean_return = top_signals['真实收益'].mean()
    
    # 4. 返回负值，因为我们要最大化收益
    return -mean_return

# --- 主函数 ---
def main():
    # 1. 加载数据
    print(f"✅ [1/4] 从缓存文件 {os.path.basename(CACHE_FILE_PATH)} 加载数据...")
    try:
        all_signals = pd.read_pickle(CACHE_FILE_PATH)
        print(f"加载了 {len(all_signals)} 条历史信号。")
    except FileNotFoundError:
        sys.exit(f"❌ 错误: 缓存文件未找到。请先运行V23分析脚本生成缓存。")
        
    # 2. 定义优化参数
    # 初始权重: 平均分配
    initial_weights = np.ones(len(FEATURES)) / len(FEATURES)
    # 权重约束: 每个权重在0到1之间，且总和为1
    bounds = [(0, 1)] * len(FEATURES)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # 3. 执行优化
    print("✅ [2/4] 正在使用SciPy优化器寻找最优特征权重...")
    print("      (这可能需要几分钟时间，优化器正在探索上千种组合...)")
    
    result = minimize(
        fun=objective_function,
        x0=initial_weights,
        args=(all_signals,),
        method='SLSQP', # 序列最小二乘规划，适合处理约束问题
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 200} # 迭代200次
    )
    
    if not result.success:
        print("⚠️ [警告] 优化器未能完全收敛，但仍会使用找到的最佳结果。")
    
    optimal_weights = result.x
    
    print("✅ [3/4] 最优权重已找到！")
    
    # 4. 分析并展示结果
    print("\n" + "="*80)
    print(" V24 特征与权重优化结果")
    print("="*80)

    # 打印最优权重
    print("\n--- 最优特征权重 ---")
    weights_df = pd.DataFrame({
        '特征': FEATURES,
        '最优权重': optimal_weights
    }).sort_values('最优权重', ascending=False)
    print(weights_df.to_string(index=False, float_format="%.4f"))

    # 使用旧权重进行对比
    print("\n--- 性能对比 ---")
    old_weights_dict = {'rsi_30': 0.25, 'bottom_depth': 0.25, 'atr_ratio': 0.15, 'rsi_14': 0.10, 'gap_open': 0.10, 'vol_amp': 0.08, 'lower_shadow_ratio': 0.07}
    old_weights = np.array([old_weights_dict.get(f, 0) for f in FEATURES])
    old_performance = -objective_function(old_weights / old_weights.sum(), all_signals)
    
    # 使用新权重计算最终性能
    new_performance = -result.fun
    
    improvement = (new_performance / old_performance - 1) * 100 if old_performance > 0 else float('inf')
    
    print(f"V23 (手动权重) Top 20% 平均收益: {old_performance:.4%}")
    print(f"V24 (优化权重) Top 20% 平均收益: {new_performance:.4%}")
    print(f"性能提升幅度: +{improvement:.2f}%")
    print("="*80)
    
    print("\n✅ [4/4] V24 特征优化完成！")

if __name__ == '__main__':
    main()
