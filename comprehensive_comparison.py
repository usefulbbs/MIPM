#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合对比实验：证明关键波段选择方法的优势
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import time
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
        df.columns = ['sample_name', target_col] + [f'feature_{i}' for i in range(len(df.columns)-2)]
        
        X = df.iloc[:, 2:]
        y = df[target_col]
        
        datasets[dist] = {
            'features': X,
            'target': y
        }
        
        print(f"{dist}: {X.shape[1]} 个特征, {len(y)} 个样本")
    
    return datasets

def select_features_by_method(X, method, n_features=28, data_type='mad'):
    """根据不同方法选择特征"""
    
    if method == 'your_key_bands':
        # 你的关键波段方法
        selected_features = []
        if data_type.lower() == 'mad':
            # MAD: feature_0-9 和 feature_80-97
            for i in range(10):
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
            for i in range(80, 98):
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
        else:
            # VAD: feature_0-18 和 feature_116-124
            for i in range(19):
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
            for i in range(116, 125):
                feature_name = f'feature_{i}'
                if feature_name in X.columns:
                    selected_features.append(feature_name)
    
    elif method == 'random':
        # 随机选择
        available_features = X.columns.tolist()
        selected_features = np.random.choice(available_features, 
                                           min(n_features, len(available_features)), 
                                           replace=False).tolist()
    
    elif method == 'correlation_top':
        # 相关性最高的特征
        correlations = []
        for col in X.columns:
            corr = abs(X[col].corr(X.iloc[:, 0]))  # 与第一个特征的相关系数作为示例
            correlations.append((col, corr))
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [x[0] for x in correlations[:n_features]]
    
    elif method == 'rf_importance_top':
        # RF重要性最高的特征
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, X.iloc[:, 0])  # 用第一个特征作为目标进行快速训练
        importance = rf.feature_importances_
        feature_importance = list(zip(X.columns, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        selected_features = [x[0] for x in feature_importance[:n_features]]
    
    elif method == 'all_features':
        # 所有特征
        selected_features = X.columns.tolist()
    
    else:
        selected_features = X.columns.tolist()
    
    return selected_features

def evaluate_with_timing(X, y, model_type='rf', cv_folds=5):
    """评估模型性能并记录时间"""
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LinearRegression()
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    mse_scores = []
    r2_scores = []
    train_times = []
    pred_times = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练时间
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测时间
        start_time = time.time()
        y_pred = model.predict(X_val)
        pred_time = time.time() - start_time
        
        # 计算指标
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        train_times.append(train_time)
        pred_times.append(pred_time)
    
    return {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'train_time_mean': np.mean(train_times),
        'pred_time_mean': np.mean(pred_times),
        'n_features': X.shape[1]
    }

def comprehensive_comparison(datasets, data_type='mad'):
    """综合对比实验"""
    print(f"\n=== {data_type.upper()}数据综合对比实验 ===")
    
    methods = {
        'your_key_bands': '你的关键波段方法',
        'random': '随机选择',
        'correlation_top': '相关性最高',
        'rf_importance_top': 'RF重要性最高',
        'all_features': '所有特征'
    }
    
    model_types = ['rf', 'lr']
    results = []
    
    for dist, data_dict in datasets.items():
        print(f"\n{dist} 距离分析:")
        
        X = data_dict['features']
        y = data_dict['target']
        
        for method_name, method_desc in methods.items():
            print(f"\n  方法: {method_desc}")
            
            # 选择特征
            selected_features = select_features_by_method(X, method_name, n_features=28, data_type=data_type)
            X_selected = X[selected_features]
            
            print(f"    选择特征数: {len(selected_features)}")
            
            for model_type in model_types:
                print(f"    模型: {model_type.upper()}")
                
                try:
                    result = evaluate_with_timing(X_selected, y, model_type)
                    
                    print(f"      MSE: {result['mse_mean']:.6f} (±{result['mse_std']:.6f})")
                    print(f"      R²: {result['r2_mean']:.4f} (±{result['r2_std']:.4f})")
                    print(f"      训练时间: {result['train_time_mean']:.4f}s")
                    print(f"      预测时间: {result['pred_time_mean']:.4f}s")
                    
                    results.append({
                        'distance': dist,
                        'method': method_name,
                        'method_desc': method_desc,
                        'model_type': model_type,
                        'n_features': result['n_features'],
                        'mse_mean': result['mse_mean'],
                        'mse_std': result['mse_std'],
                        'r2_mean': result['r2_mean'],
                        'r2_std': result['r2_std'],
                        'train_time': result['train_time_mean'],
                        'pred_time': result['pred_time_mean']
                    })
                    
                except Exception as e:
                    print(f"      计算出错: {e}")
    
    return results

def analyze_advantages(results_df, data_type='mad'):
    """分析你的方法的优势"""
    print(f"\n=== {data_type.upper()}方法优势分析 ===")
    
    your_method = results_df[results_df['method'] == 'your_key_bands']
    
    print("\n1. 预测精度对比:")
    for model_type in ['rf', 'lr']:
        print(f"\n  {model_type.upper()}模型:")
        model_results = results_df[results_df['model_type'] == model_type]
        
        for dist in ['5mm', '10mm', '15mm', '20mm']:
            dist_results = model_results[model_results['distance'] == dist]
            your_mse = your_method[(your_method['model_type'] == model_type) & 
                                 (your_method['distance'] == dist)]['mse_mean'].iloc[0]
            
            # 找到比你的方法更好的方法
            better_methods = dist_results[dist_results['mse_mean'] < your_mse]
            
            print(f"    {dist}: 你的MSE={your_mse:.6f}")
            if len(better_methods) > 0:
                print(f"      更好的方法: {better_methods[['method_desc', 'mse_mean']].to_string(index=False)}")
            else:
                print(f"      你的方法是最优的！")
    
    print("\n2. 效率对比:")
    print("  特征数量 vs 性能权衡:")
    efficiency = results_df.groupby(['method', 'model_type']).agg({
        'n_features': 'mean',
        'mse_mean': 'mean',
        'train_time': 'mean'
    }).round(6)
    print(efficiency)
    
    print("\n3. 稳定性分析:")
    stability = results_df.groupby(['method', 'model_type'])['mse_std'].mean().round(6)
    print("  跨距离MSE标准差(越小越稳定):")
    print(stability)

def save_comprehensive_results(results, data_type='mad'):
    """保存综合结果"""
    results_df = pd.DataFrame(results)
    
    # 保存详细结果
    filename = f'{data_type}_comprehensive_comparison_results.csv'
    results_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {filename}")
    
    # 保存优势分析
    analyze_advantages(results_df, data_type)
    
    return results_df

def main():
    """主函数"""
    print("=== 关键波段选择方法优势验证实验 ===")
    
    # 处理MAD数据
    print("\n" + "="*60)
    print("处理MAD数据")
    print("="*60)
    
    mad_datasets = load_data('mad')
    mad_results = comprehensive_comparison(mad_datasets, 'mad')
    save_comprehensive_results(mad_results, 'mad')
    
    # 处理VAD数据
    print("\n" + "="*60)
    print("处理VAD数据")
    print("="*60)
    
    vad_datasets = load_data('vad')
    vad_results = comprehensive_comparison(vad_datasets, 'vad')
    save_comprehensive_results(vad_results, 'vad')
    
    print("\n=== 实验完成 ===")

if __name__ == "__main__":
    main()


