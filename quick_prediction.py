#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于现有mse_comparison.py的预测流程，直接替换特征选择
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data(data_type='mad'):
    """加载数据"""
    print(f"正在加载{data_type.upper()}数据...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    datasets = {}
    
    for dist in distances:
        if data_type.lower() == 'mad': 
            file_path = f'dataset_mad/mad_{dist}.xlsx'
            target_col = 'Mad'
        else:
            file_path = f'dataset_vad/vad_{dist}.xlsx'
            target_col = 'Vad'
            
        df = pd.read_excel(file_path)
        
        # 重命名列
        df.columns = ['sample_name', target_col] + [f'feature_{i}' for i in range(len(df.columns)-2)]
        
        # 分离特征和目标变量
        X = df.iloc[:, 2:]  # 特征列
        y = df[target_col]   # 目标变量
        
        datasets[dist] = {
            'features': X,
            'target': y,
            'feature_names': X.columns.tolist()
        }
        
        print(f"{dist}: {X.shape[1]} 个特征, {len(y)} 个样本")
    
    return datasets

def select_specific_features(datasets, data_type='mad'):
    """选择指定的特征"""
    print(f"\n选择{data_type.upper()}指定特征...")
    
    selected_datasets = {}
    
    for dist, data_dict in datasets.items():
        X = data_dict['features']
        y = data_dict['target']
        
        # 构建特征选择列表
        selected_features = []
        
        if data_type.lower() == 'mad':
            # MAD: feature_0到feature_9和feature_80到feature_97
            for i in range(10):  # 0到9
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
            
            for i in range(80, 98):  # 80到97
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
        else:
            # VAD: feature_0到feature_18和feature_116到feature_124
            for i in range(19):  # 0到18
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
            
            for i in range(116, 125):  # 116到124
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
        
        print(f"{dist} 距离选择的特征数: {len(selected_features)}")
        
        if len(selected_features) == 0:
            print(f"  警告: {dist} 距离没有找到指定特征")
            continue
        
        # 选择特征子集
        X_selected = X[selected_features]
        
        selected_datasets[dist] = {
            'features': X_selected,
            'target': y,
            'selected_features': selected_features
        }
    
    return selected_datasets

def evaluate_model_performance(X, y, model_type='rf', cv_folds=5):
    """评估模型性能 - 直接复制mse_comparison.py的逻辑"""
    # 创建模型
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LinearRegression()
    
    # 交叉验证
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    return mean_mse, std_mse

def predict_with_specific_features(datasets, data_type='mad'):
    """使用指定特征预测"""
    print(f"\n=== 使用指定特征预测{data_type.upper()}标签并计算MSE ===")
    
    results = []
    model_types = ['rf', 'lr']
    
    for dist, data_dict in datasets.items():
        print(f"\n{dist} 距离分析:")
        
        X = data_dict['features']
        y = data_dict['target']
        
        print(f"  特征数量: {X.shape[1]}")
        print(f"  样本数量: {len(y)}")
        print(f"  {data_type.upper()}值范围: {y.min():.4f} ~ {y.max():.4f}")
        print(f"  {data_type.upper()}均值: {y.mean():.4f}")
        
        for model_type in model_types:
            print(f"\n  使用 {model_type.upper()} 模型:")
            
            try:
                mean_mse, std_mse = evaluate_model_performance(X, y, model_type, cv_folds=5)
                
                print(f"    交叉验证MSE: {mean_mse:.6f} (±{std_mse:.6f})")
                
                results.append({
                    'distance': dist,
                    'model_type': model_type,
                    'n_features': X.shape[1],
                    'mse': mean_mse,
                    'std': std_mse
                })
                
            except Exception as e:
                print(f"    计算出错: {e}")
                results.append({
                    'distance': dist,
                    'model_type': model_type,
                    'n_features': X.shape[1],
                    'mse': np.inf,
                    'std': np.inf,
                    'error': str(e)
                })
    
    return results

def save_results(results, data_type='mad'):
    """保存结果"""
    print(f"\n=== 保存{data_type.upper()}结果 ===")
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存CSV
    filename = f'{data_type}_specific_features_mse_results.csv'
    results_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {filename}")
    
    # 打印汇总表
    print(f"\n=== {data_type.upper()}结果汇总表 ===")
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    return results_df

def main():
    """主函数"""
    print("=== 使用指定特征预测MAD和VAD标签 ===")
    
    # 处理MAD数据
    print("\n" + "="*50)
    print("处理MAD数据")
    print("="*50)
    
    mad_datasets = load_data('mad')
    mad_selected = select_specific_features(mad_datasets, 'mad')
    if mad_selected:
        mad_results = predict_with_specific_features(mad_selected, 'mad')
        save_results(mad_results, 'mad')
    
    # 处理VAD数据
    print("\n" + "="*50)
    print("处理VAD数据")
    print("="*50)
    
    vad_datasets = load_data('vad')
    vad_selected = select_specific_features(vad_datasets, 'vad')
    if vad_selected:
        vad_results = predict_with_specific_features(vad_selected, 'vad')
        save_results(vad_results, 'vad')
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
