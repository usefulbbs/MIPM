#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用指定特征预测MAD标签并计算MSE
特征范围：feature_0 到 feature_9 和 feature_80 到 feature_97
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_mad_data():
    """加载四个距离的MAD数据"""
    print("正在加载MAD数据...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    datasets = {}
    
    for dist in distances:
        file_path = f'dataset_mad/mad_{dist}.xlsx'
        df = pd.read_excel(file_path)
        
        # 重命名列
        df.columns = ['sample_name', 'Mad'] + [f'feature_{i}' for i in range(len(df.columns)-2)]
        
        # 分离特征和目标变量
        X = df.iloc[:, 2:]  # 特征列
        y = df['Mad']        # 目标变量
        
        datasets[dist] = {
            'data': df,
            'features': X,
            'target': y,
            'feature_names': X.columns.tolist()
        }
        
        print(f"{dist}: {X.shape[1]} 个特征, {len(y)} 个样本")
    
    return datasets

def select_specific_features(datasets):
    """选择指定的特征：feature_0到feature_9和feature_80到feature_97"""
    print("\n选择指定特征...")
    
    selected_datasets = {}
    
    for dist, data_dict in datasets.items():
        X = data_dict['features']
        y = data_dict['target']
        
        # 构建特征选择列表
        selected_features = []
        
        # 添加feature_0到feature_9
        for i in range(10):
            feature_name = f'feature_{i}'
            if feature_name in X.columns:
                selected_features.append(feature_name)
        
        # 添加feature_80到feature_97
        for i in range(80, 98):
            feature_name = f'feature_{i}'
            if feature_name in X.columns:
                selected_features.append(feature_name)
        
        print(f"{dist} 距离选择的特征:")
        print(f"  特征0-9: {[f for f in selected_features if f.startswith('feature_') and int(f.split('_')[1]) < 10]}")
        print(f"  特征80-97: {[f for f in selected_features if f.startswith('feature_') and 80 <= int(f.split('_')[1]) <= 97]}")
        print(f"  总特征数: {len(selected_features)}")
        
        # 选择特征子集
        X_selected = X[selected_features]
        
        selected_datasets[dist] = {
            'features': X_selected,
            'target': y,
            'selected_features': selected_features,
            'feature_names': selected_features
        }
    
    return selected_datasets

def evaluate_with_models(datasets):
    """使用不同模型评估性能"""
    print("\n=== 使用指定特征预测MAD标签并计算MSE ===")
    
    results = {}
    
    # 定义模型
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    for dist, data_dict in datasets.items():
        print(f"\n{dist} 距离分析:")
        
        X = data_dict['features']
        y = data_dict['target']
        
        print(f"  特征数量: {X.shape[1]}")
        print(f"  样本数量: {len(y)}")
        print(f"  MAD值范围: {y.min():.4f} ~ {y.max():.4f}")
        print(f"  MAD均值: {y.mean():.4f}")
        
        dist_results = {}
        
        for model_name, model in models.items():
            print(f"\n  使用 {model_name}:")
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 5折交叉验证计算MSE
            try:
                mse_scores = -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
                mean_mse = mse_scores.mean()
                std_mse = mse_scores.std()
                
                # 计算R²分数
                r2_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                mean_r2 = r2_scores.mean()
                std_r2 = r2_scores.std()
                
                print(f"    交叉验证MSE: {mean_mse:.6f} (±{std_mse:.6f})")
                print(f"    交叉验证R²: {mean_r2:.4f} (±{std_r2:.4f})")
                
                dist_results[model_name] = {
                    'mse_mean': mean_mse,
                    'mse_std': std_mse,
                    'r2_mean': mean_r2,
                    'r2_std': std_r2,
                    'mse_scores': mse_scores,
                    'r2_scores': r2_scores
                }
                
            except Exception as e:
                print(f"    计算出错: {e}")
                dist_results[model_name] = {
                    'mse_mean': np.inf,
                    'mse_std': np.inf,
                    'r2_mean': -np.inf,
                    'r2_std': np.inf,
                    'error': str(e)
                }
        
        results[dist] = dist_results
    
    return results

def create_summary_table(results):
    """创建汇总表"""
    print("\n=== 结果汇总表 ===")
    
    summary_data = []
    
    for dist in ['5mm', '10mm', '15mm', '20mm']:
        if dist in results:
            for model_name in ['Random Forest', 'Linear Regression']:
                if model_name in results[dist]:
                    result = results[dist][model_name]
                    summary_data.append({
                        '距离': dist,
                        '模型': model_name,
                        'MSE均值': result['mse_mean'],
                        'MSE标准差': result['mse_std'],
                        'R²均值': result['r2_mean'],
                        'R²标准差': result['r2_std']
                    })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format='%.6f'))
    
    return summary_df

def save_results(results, summary_df):
    """保存结果"""
    print("\n=== 保存结果 ===")
    
    # 保存汇总表
    summary_df.to_csv('specific_features_prediction_results.csv', index=False, encoding='utf-8-sig')
    print("汇总表已保存到: specific_features_prediction_results.csv")
    
    # 保存详细结果
    detailed_results = []
    for dist, dist_results in results.items():
        for model_name, model_results in dist_results.items():
            if 'mse_scores' in model_results:
                for i, (mse, r2) in enumerate(zip(model_results['mse_scores'], model_results['r2_scores'])):
                    detailed_results.append({
                        '距离': dist,
                        '模型': model_name,
                        '折数': i+1,
                        'MSE': mse,
                        'R²': r2
                    })
    
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('specific_features_detailed_results.csv', index=False, encoding='utf-8-sig')
        print("详细结果已保存到: specific_features_detailed_results.csv")

def main():
    """主函数"""
    print("=== 使用指定特征预测MAD标签 ===")
    print("特征范围: feature_0 到 feature_9 和 feature_80 到 feature_97")
    
    # 1. 加载数据
    datasets = load_mad_data()
    
    # 2. 选择指定特征
    selected_datasets = select_specific_features(datasets)
    
    # 3. 评估模型性能
    results = evaluate_with_models(selected_datasets)
    
    # 4. 创建汇总表
    summary_df = create_summary_table(results)
    
    # 5. 保存结果
    save_results(results, summary_df)
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
