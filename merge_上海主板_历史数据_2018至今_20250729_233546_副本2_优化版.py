#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并优化分割的CSV文件 (已删除股票名称列)
使用方法: python merge_上海主板_历史数据_2018至今_20250729_233546_副本2_优化版.py
"""

import pandas as pd
import os

def merge_csv_files():
    """合并分割的CSV文件"""
    print("开始合并优化文件...")
    
    # 读取所有分割文件
    dfs = []
    for i in range(1, 17):
        file_path = f"上海主板_历史数据_2018至今_20250729_233546_副本2_优化版_part_{i:02d}.csv"
        if os.path.exists(file_path):
            print(f"读取文件: {file_path}")
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"警告: 文件不存在 {file_path}")
    
    if not dfs:
        print("错误: 没有找到任何分割文件")
        return
    
    # 合并数据
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"合并完成，总行数: {len(merged_df):,}")
    
    # 保存合并后的文件
    output_file = f"上海主板_历史数据_2018至今_20250729_233546_副本2_优化版_merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"已保存合并文件: {output_file}")
    
    # 显示文件大小
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"合并文件大小: {file_size:.2f} MB")
    print("注意: 此文件已删除股票名称列，只保留股票代码")

if __name__ == "__main__":
    merge_csv_files()
