import pandas as pd
import numpy as np
import time
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (V31 AI特征学习版)
# ==============================================================================
# 上海+深圳数据路径
SHANGHAI_DATA_PATH = '/Users/yamijin/Desktop/A股主板历史数据_2018至今_20250729_233546/上海主板_历史数据_2018至今_20250729_233546.csv'
SHENZHEN_DATA_PATH = '/Users/yamijin/Desktop/A股主板历史数据_2018至今_20250729_233546/深圳主板_历史数据_2018至今_20250729_233546.csv'

OUTPUT_MODEL_PATH = 'V31_AI_Feature_Model.pkl'
OUTPUT_SCALER_PATH = 'V31_AI_Feature_Scaler.pkl'
OUTPUT_TRADES_PATH = 'V31_AI_Enhanced_Trades.csv'

# V6核心参数
V_BOTTOM_LOOKBACK = 20
STOP_LOSS_PCT = 0.031
MAX_HOLDING_DAYS = 10
FEATURE_WINDOW = 4  # 使用4天窗口提取特征

# ==============================================================================
# 数据加载与预处理
# ==============================================================================
def load_combined_data():
    """加载上海+深圳数据并合并"""
    print("开始加载上海+深圳数据...")
    start_time = time.time()
    
    # 加载两个市场数据
    try:
        sh_data = pd.read_csv(SHANGHAI_DATA_PATH)
        sz_data = pd.read_csv(SHENZHEN_DATA_PATH)
        print(f"上海数据: {len(sh_data):,} 条")
        print(f"深圳数据: {len(sz_data):,} 条")
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        return None
    
    # 合并数据
    all_data = pd.concat([sh_data, sz_data], ignore_index=True)
    
    # 标准化列名
    all_data.rename(columns={
        'ts_code': '代码', 'trade_date': '日期', 'open': '开盘',
        'high': '最高', 'low': '最低', 'close': '收盘',
        'vol': '成交量', 'amount': '成交额'
    }, inplace=True)
    
    all_data['日期'] = pd.to_datetime(all_data['日期'], format='%Y%m%d')
    all_data.sort_values(by=['代码', '日期'], inplace=True)
    
    print(f"合并后总数据: {len(all_data):,} 条，{all_data['代码'].nunique()} 只股票")
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f} 秒")
    return all_data

# ==============================================================================
# 高级特征提取 (4天窗口)
# ==============================================================================
def extract_advanced_features(stock_data, signal_idx):
    """
    提取信号点的高级特征 (使用4天窗口: 当天+前3天)
    """
    try:
        if signal_idx < FEATURE_WINDOW:
            return None
            
        # 获取4天窗口数据
        window_data = stock_data.iloc[signal_idx-FEATURE_WINDOW+1:signal_idx+1]
        if len(window_data) != FEATURE_WINDOW:
            return None
            
        features = {}
        
        # 1. 基础价格特征 (4天)
        opens = window_data['开盘'].values
        highs = window_data['最高'].values
        lows = window_data['最低'].values
        closes = window_data['收盘'].values
        volumes = window_data['成交量'].values
        amounts = window_data['成交额'].values if '成交额' in window_data.columns else volumes * closes
        
        # 2. 价格动量特征
        features['price_momentum_3d'] = (closes[-1] - closes[0]) / closes[0]  # 3天总涨幅
        features['price_acceleration'] = closes[-1] - 2*closes[-2] + closes[-3]  # 价格加速度
        features['consecutive_up_strength'] = np.sum(np.diff(closes) > 0)  # 连续上涨强度
        
        # 3. 波动率特征
        returns = np.diff(closes) / closes[:-1]
        features['volatility_4d'] = np.std(returns)  # 4天波动率
        features['max_intraday_range'] = np.max((highs - lows) / closes)  # 最大日内波动
        features['price_stability'] = 1 / (1 + features['volatility_4d'])  # 价格稳定性
        
        # 4. 成交量特征
        features['volume_trend'] = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
        features['volume_acceleration'] = volumes[-1] - 2*volumes[-2] + volumes[-3]
        features['avg_volume_4d'] = np.mean(volumes)
        features['volume_consistency'] = 1 / (1 + np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0
        
        # 5. 成交额特征
        features['amount_growth'] = (amounts[-1] - amounts[0]) / amounts[0] if amounts[0] > 0 else 0
        features['amount_volume_ratio'] = np.mean(amounts / volumes) if np.all(volumes > 0) else 0
        
        # 6. K线形态特征 (最后一天)
        last_day = window_data.iloc[-1]
        body_size = abs(last_day['收盘'] - last_day['开盘'])
        total_range = last_day['最高'] - last_day['最低']
        features['body_ratio'] = body_size / total_range if total_range > 0 else 0
        features['upper_shadow'] = (last_day['最高'] - max(last_day['开盘'], last_day['收盘'])) / total_range if total_range > 0 else 0
        features['lower_shadow'] = (min(last_day['开盘'], last_day['收盘']) - last_day['最低']) / total_range if total_range > 0 else 0
        
        # 7. 相对位置特征
        features['close_vs_high_4d'] = closes[-1] / np.max(highs)  # 收盘价相对4天最高价
        features['close_vs_low_4d'] = closes[-1] / np.min(lows)   # 收盘价相对4天最低价
        features['position_in_range'] = (closes[-1] - np.min(lows)) / (np.max(highs) - np.min(lows)) if np.max(highs) > np.min(lows) else 0.5
        
        # 8. 趋势强度特征
        # 计算简单移动平均趋势
        if signal_idx >= 10:
            ma5_current = stock_data.iloc[signal_idx-4:signal_idx+1]['收盘'].mean()
            ma5_prev = stock_data.iloc[signal_idx-9:signal_idx-4]['收盘'].mean()
            features['ma5_trend'] = (ma5_current - ma5_prev) / ma5_prev if ma5_prev > 0 else 0
        else:
            features['ma5_trend'] = 0
            
        # 9. V型底深度特征 (基于V6逻辑)
        if signal_idx >= V_BOTTOM_LOOKBACK + 3:
            day0_idx = signal_idx - 3  # 第0天位置
            window_start = max(0, day0_idx - V_BOTTOM_LOOKBACK + 1)
            v_window_prices = stock_data.iloc[window_start:day0_idx+1]['收盘'].values
            if len(v_window_prices) > 0:
                min_price = np.min(v_window_prices)
                max_price = np.max(v_window_prices)
                features['v_bottom_depth'] = (max_price - min_price) / max_price if max_price > 0 else 0
                features['v_recovery_ratio'] = (closes[-1] - min_price) / (max_price - min_price) if max_price > min_price else 0
            else:
                features['v_bottom_depth'] = 0
                features['v_recovery_ratio'] = 0
        else:
            features['v_bottom_depth'] = 0
            features['v_recovery_ratio'] = 0
            
        # 10. 市场微观结构特征
        features['price_gaps'] = np.sum(opens[1:] != closes[:-1])  # 跳空次数
        features['closing_strength'] = np.mean((closes - lows) / (highs - lows))  # 收盘强度
        
        return features
        
    except Exception as e:
        print(f"特征提取错误: {e}")
        return None

# ==============================================================================
# V6信号检测与特征标签生成
# ==============================================================================
def generate_training_data(all_data):
    """生成训练数据：特征+标签(实际收益率)"""
    print("开始生成训练数据...")
    start_time = time.time()
    
    training_samples = []
    grouped = all_data.groupby('代码')
    
    for code, stock_data in grouped:
        if len(stock_data) < V_BOTTOM_LOOKBACK + 10:
            continue
            
        df = stock_data.copy().reset_index(drop=True)
        closes = df['收盘'].values
        
        # 寻找V6信号点
        for i in range(V_BOTTOM_LOOKBACK + 4, len(closes) - MAX_HOLDING_DAYS):
            # V6条件1: 连续3天不减
            if not (closes[i-2] <= closes[i-1] <= closes[i]):
                continue
                
            # V6条件2: V型底
            day0_idx = i - 3
            window_start = max(0, day0_idx - V_BOTTOM_LOOKBACK + 1)
            window_closes = closes[window_start:day0_idx + 1]
            if closes[day0_idx] != min(window_closes):
                continue
                
            # 提取特征
            features = extract_advanced_features(df, i)
            if not features:
                continue
                
            # 模拟交易计算实际收益 (简化版V6逻辑)
            entry_price = closes[i]
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            
            # 寻找未来收益
            actual_return = 0
            for j in range(1, min(MAX_HOLDING_DAYS + 1, len(closes) - i)):
                future_close = closes[i + j]
                if future_close < stop_loss_price:
                    actual_return = (stop_loss_price - entry_price) / entry_price
                    break
                elif j == MAX_HOLDING_DAYS:
                    actual_return = (future_close - entry_price) / entry_price
                    
            # 添加到训练样本
            sample = features.copy()
            sample['actual_return'] = actual_return
            sample['stock_code'] = code
            sample['signal_date'] = df.iloc[i]['日期']
            training_samples.append(sample)
    
    training_df = pd.DataFrame(training_samples)
    print(f"训练数据生成完成: {len(training_df)} 个样本，耗时: {time.time() - start_time:.2f} 秒")
    return training_df

# ==============================================================================
# AI模型训练
# ==============================================================================
def train_ai_model(training_df):
    """训练AI特征学习模型"""
    print("开始训练AI模型...")
    start_time = time.time()
    
    # 准备特征和标签
    feature_cols = [col for col in training_df.columns if col not in ['actual_return', 'stock_code', 'signal_date']]
    X = training_df[feature_cols].fillna(0)
    y = training_df['actual_return']
    
    print(f"特征维度: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 训练随机森林模型
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 模型评估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    print(f"\n--- AI模型训练结果 ---")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n--- Top 10 重要特征 ---")
    print(feature_importance.head(10))
    
    # 保存模型
    joblib.dump(model, OUTPUT_MODEL_PATH)
    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f} 秒")
    return model, scaler, feature_cols

# ==============================================================================
# AI增强回测
# ==============================================================================
def run_ai_enhanced_backtest(all_data, model, scaler, feature_cols, top_percentile=0.3):
    """使用AI模型进行增强回测"""
    print(f"开始AI增强回测 (Top {top_percentile*100:.0f}%)...")
    start_time = time.time()
    
    # 生成所有信号和AI评分
    all_signals = []
    grouped = all_data.groupby('代码')
    
    for code, stock_data in grouped:
        if len(stock_data) < V_BOTTOM_LOOKBACK + 10:
            continue
            
        df = stock_data.copy().reset_index(drop=True)
        closes = df['收盘'].values
        
        for i in range(V_BOTTOM_LOOKBACK + 4, len(closes) - 2):
            # V6信号检测
            if not (closes[i-2] <= closes[i-1] <= closes[i]):
                continue
                
            day0_idx = i - 3
            window_start = max(0, day0_idx - V_BOTTOM_LOOKBACK + 1)
            window_closes = closes[window_start:day0_idx + 1]
            if closes[day0_idx] != min(window_closes):
                continue
                
            # 提取特征并预测
            features = extract_advanced_features(df, i)
            if not features:
                continue
                
            feature_vector = np.array([[features.get(col, 0) for col in feature_cols]])
            feature_scaled = scaler.transform(feature_vector)
            ai_score = model.predict(feature_scaled)[0]
            
            all_signals.append({
                '代码': code,
                '日期': df.iloc[i]['日期'],
                '买入价格': closes[i],
                'AI评分': ai_score
            })
    
    signals_df = pd.DataFrame(all_signals)
    print(f"生成 {len(signals_df)} 个AI评分信号")
    
    if signals_df.empty:
        return pd.DataFrame()
    
    # 筛选Top信号
    threshold = signals_df['AI评分'].quantile(1 - top_percentile)
    elite_signals = signals_df[signals_df['AI评分'] >= threshold].copy()
    
    print(f"筛选出 {len(elite_signals)} 个精英信号 (阈值: {threshold:.4f})")
    
    # 执行回测 (简化版)
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
                
            # 简化交易模拟
            entry_price = signal['买入价格']
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            
            exit_price = None
            exit_date = None
            exit_reason = "持有到期"
            
            for i in range(min(len(future_data), MAX_HOLDING_DAYS)):
                current_close = future_data.iloc[i]['收盘']
                if current_close < stop_loss_price:
                    exit_price = stop_loss_price
                    exit_date = future_data.iloc[i]['日期']
                    exit_reason = "止损"
                    break
                    
            if exit_price is None:
                exit_idx = min(MAX_HOLDING_DAYS - 1, len(future_data) - 1)
                exit_price = future_data.iloc[exit_idx]['收盘']
                exit_date = future_data.iloc[exit_idx]['日期']
            
            total_return = (exit_price - entry_price) / entry_price
            
            trades.append({
                '代码': stock_code,
                '买入日期': current_date,
                '买入价格': entry_price,
                '卖出日期': exit_date,
                '卖出价格': exit_price,
                '收益率': total_return,
                '退出原因': exit_reason,
                'AI评分': signal['AI评分']
            })
            
            active_trades[stock_code] = exit_date
    
    trades_df = pd.DataFrame(trades)
    print(f"AI增强回测完成: {len(trades_df)} 笔交易，耗时: {time.time() - start_time:.2f} 秒")
    return trades_df

# ==============================================================================
# 性能分析
# ==============================================================================
def analyze_ai_performance(trades_df):
    """分析AI增强策略性能"""
    if trades_df.empty:
        print("AI回测未产生任何交易")
        return
        
    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['收益率'] > 0]
    win_rate = len(win_trades) / total_trades
    avg_return = trades_df['收益率'].mean()
    avg_win = win_trades['收益率'].mean() if not win_trades.empty else 0
    avg_loss = trades_df[trades_df['收益率'] <= 0]['收益率'].mean()
    
    print("\n" + "="*70)
    print("V31 AI特征学习策略性能分析")
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
    print(f"V31 AI增强: {avg_return:.4%}")
    
    if avg_return >= 0.02:
        print("🎉 突破2%目标！")
    elif avg_return > 0.0189:
        print("✅ 超越V6基准！")
    else:
        print("❌ 需要进一步优化")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    # 1. 加载数据
    all_data = load_combined_data()
    if all_data is None:
        return
        
    # 2. 生成训练数据
    training_df = generate_training_data(all_data)
    if training_df.empty:
        print("训练数据生成失败")
        return
        
    # 3. 训练AI模型
    model, scaler, feature_cols = train_ai_model(training_df)
    
    # 4. AI增强回测
    trades_df = run_ai_enhanced_backtest(all_data, model, scaler, feature_cols)
    
    # 5. 保存和分析结果
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_TRADES_PATH, index=False, encoding='utf-8-sig')
        analyze_ai_performance(trades_df)
    
    print(f"\n模型已保存: {OUTPUT_MODEL_PATH}")
    print(f"标准化器已保存: {OUTPUT_SCALER_PATH}")

if __name__ == '__main__':
    main()