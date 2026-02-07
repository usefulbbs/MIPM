import pandas as pd
import numpy as np

def check_mad_data():
    """检查MAD数据集的特征数量差异"""
    print("=== 检查MAD数据集特征数量差异 ===")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    
    for dist in distances:
        try:
            file_path = f'dataset_mad/mad_{dist}.xlsx'
            df = pd.read_excel(file_path)
            
            # 计算特征数量（排除sample_name和Mad列）
            feature_count = len(df.columns) - 2
            sample_count = len(df)
            
            print(f"\n{dist} 距离:")
            print(f"  总列数: {len(df.columns)}")
            print(f"  特征数量: {feature_count}")
            print(f"  样本数量: {sample_count}")
            
            # 检查前几列的名称
            print(f"  列名: {list(df.columns[:5])}")
            
            # 检查是否有缺失值
            missing_count = df.isnull().sum().sum()
            print(f"  缺失值总数: {missing_count}")
            
        except Exception as e:
            print(f"\n{dist} 距离数据读取失败: {e}")

if __name__ == "__main__":
    check_mad_data()