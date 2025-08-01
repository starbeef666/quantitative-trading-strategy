#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版分割大文件脚本
删除股票名称列，只保留股票代码，大幅减少文件大小
"""

import pandas as pd
import os
import math

def split_csv_file_optimized(input_file, output_prefix, max_size_mb=20):
    """
    优化版分割CSV文件 - 删除股票名称列
    
    Args:
        input_file: 输入文件路径
        output_prefix: 输出文件前缀
        max_size_mb: 每个文件最大大小(MB)
    """
    print(f"开始优化分割文件: {input_file}")
    
    # 读取文件大小
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"原始文件大小: {file_size:.2f} MB")
    
    # 读取CSV文件
    print("正在读取CSV文件...")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    print(f"总行数: {total_rows:,}")
    
    # 删除股票名称列
    if '股票名称' in df.columns:
        df = df.drop(columns=['股票名称'])
        print("已删除股票名称列")
    
    # 重新计算文件大小
    temp_file = f"{output_prefix}_temp.csv"
    df.to_csv(temp_file, index=False)
    optimized_size = os.path.getsize(temp_file) / (1024 * 1024)  # MB
    print(f"优化后文件大小: {optimized_size:.2f} MB")
    print(f"节省空间: {file_size - optimized_size:.2f} MB ({((file_size - optimized_size) / file_size * 100):.1f}%)")
    
    # 计算需要分割的文件数量
    num_parts = math.ceil(optimized_size / max_size_mb)
    print(f"需要分割成 {num_parts} 个文件")
    
    rows_per_file = math.ceil(total_rows / num_parts)
    print(f"每个文件行数: {rows_per_file:,}")
    
    # 分割文件
    for i in range(num_parts):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)
        
        # 提取数据
        part_df = df.iloc[start_idx:end_idx]
        
        # 保存文件
        output_file = f"{output_prefix}_part_{i+1:02d}.csv"
        part_df.to_csv(output_file, index=False)
        
        # 计算文件大小
        part_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"已创建: {output_file} ({len(part_df):,} 行, {part_size:.2f} MB)")
    
    # 删除临时文件
    os.remove(temp_file)
    
    print(f"优化分割完成！共创建 {num_parts} 个文件")
    return num_parts

def create_merge_script_optimized(output_prefix, num_parts):
    """创建优化版合并脚本"""
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并优化分割的CSV文件 (已删除股票名称列)
使用方法: python merge_{output_prefix}.py
"""

import pandas as pd
import os

def merge_csv_files():
    """合并分割的CSV文件"""
    print("开始合并优化文件...")
    
    # 读取所有分割文件
    dfs = []
    for i in range(1, {num_parts + 1}):
        file_path = f"{output_prefix}_part_{{i:02d}}.csv"
        if os.path.exists(file_path):
            print(f"读取文件: {{file_path}}")
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"警告: 文件不存在 {{file_path}}")
    
    if not dfs:
        print("错误: 没有找到任何分割文件")
        return
    
    # 合并数据
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"合并完成，总行数: {{len(merged_df):,}}")
    
    # 保存合并后的文件
    output_file = f"{output_prefix}_merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"已保存合并文件: {{output_file}}")
    
    # 显示文件大小
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"合并文件大小: {{file_size:.2f}} MB")
    print("注意: 此文件已删除股票名称列，只保留股票代码")

if __name__ == "__main__":
    merge_csv_files()
'''
    
    script_file = f"merge_{output_prefix}.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"已创建优化合并脚本: {script_file}")

def main():
    """主函数"""
    print("=" * 60)
    print("优化版大文件分割工具 - 删除股票名称列")
    print("=" * 60)
    
    # 分割上海数据
    print("\n1. 优化分割上海数据...")
    shanghai_parts = split_csv_file_optimized(
        "/Users/yamijin/Desktop/上海主板_历史数据_2018至今_20250729_233546_副本2.csv",
        "上海主板_历史数据_2018至今_20250729_233546_副本2_优化版",
        max_size_mb=20
    )
    create_merge_script_optimized("上海主板_历史数据_2018至今_20250729_233546_副本2_优化版", shanghai_parts)
    
    # 分割深圳数据
    print("\n2. 优化分割深圳数据...")
    shenzhen_parts = split_csv_file_optimized(
        "/Users/yamijin/Desktop/深圳主板_历史数据_2018至今_20250729_233546_副本.csv",
        "深圳主板_历史数据_2018至今_20250729_233546_副本_优化版",
        max_size_mb=20
    )
    create_merge_script_optimized("深圳主板_历史数据_2018至今_20250729_233546_副本_优化版", shenzhen_parts)
    
    print("\n" + "=" * 60)
    print("优化分割完成！")
    print("=" * 60)
    print("\n📋 优化效果:")
    print(f"- 上海数据: 分割成 {shanghai_parts} 个文件")
    print(f"- 深圳数据: 分割成 {shenzhen_parts} 个文件")
    print("- 已删除股票名称列，大幅减少文件大小")
    print("\n📁 需要上传的文件:")
    print("- 上海主板_历史数据_2018至今_20250729_233546_副本2_优化版_part_*.csv")
    print("- 深圳主板_历史数据_2018至今_20250729_233546_副本_优化版_part_*.csv")
    print("- merge_*_优化版.py 脚本")

if __name__ == "__main__":
    main() 