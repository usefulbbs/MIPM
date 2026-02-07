#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版本：使用指定特征预测MAD标签并计算MSE
避免复杂的库依赖问题
"""

import pandas as pd
import numpy as np
import os

def load_mad_data():
    """加载四个距离的MAD数据"""
    print("正在加载MAD数据...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    datasets = {}
    
    for dist in distances:
        file_path = f'dataset_mad/mad_{dist}.xlsx'
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            continue
            
        try:
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
            
        except Exception as e:
            print(f"加载 {dist} 数据时出错: {e}")
            continue
    
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
        
        if len(selected_features) == 0:
            print(f"  警告: {dist} 距离没有找到指定特征")
            continue
        
        # 选择特征子集
        X_selected = X[selected_features]
        
        selected_datasets[dist] = {
            'features': X_selected,
            'target': y,
            'selected_features': selected_features,
            'feature_names': selected_features
        }
    
    return selected_datasets

def simple_linear_regression(X, y):
    """简单的线性回归实现"""
    # 添加截距项
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    
    # 计算最小二乘解
    try:
        # (X'X)^(-1)X'y
        XtX = np.dot(X_with_intercept.T, X_with_intercept)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X_with_intercept.T, y)
        coefficients = np.dot(XtX_inv, Xty)
        
        # 预测
        y_pred = np.dot(X_with_intercept, coefficients)
        
        # 计算MSE
        mse = np.mean((y - y_pred) ** 2)
        
        # 计算R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return mse, r2, y_pred
        
    except np.linalg.LinAlgError:
        return np.inf, -np.inf, None

def cross_validation_mse(X, y, k_folds=5):
    """简单的交叉验证"""
    n_samples = len(y)
    fold_size = n_samples // k_folds
    
    mse_scores = []
    r2_scores = []
    
    for i in range(k_folds):
        # 划分训练集和验证集
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k_folds - 1 else n_samples
        
        val_indices = list(range(start_idx, end_idx))
        train_indices = [j for j in range(n_samples) if j not in val_indices]
        
        X_train = X.iloc[train_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]
        
        # 训练模型
        mse, r2, _ = simple_linear_regression(X_train.values, y_train.values)
        
        if mse != np.inf:
            # 在验证集上预测
            X_val_with_intercept = np.column_stack([np.ones(X_val.shape[0]), X_val.values])
            
            # 重新训练以获取系数
            X_train_with_intercept = np.column_stack([np.ones(X_train.shape[0]), X_train.values])
            XtX = np.dot(X_train_with_intercept.T, X_train_with_intercept)
            XtX_inv = np.linalg.inv(XtX)
            Xty = np.dot(X_train_with_intercept.T, y_train.values)
            coefficients = np.dot(XtX_inv, Xty)
            
            y_val_pred = np.dot(X_val_with_intercept, coefficients)
            val_mse = np.mean((y_val.values - y_val_pred) ** 2)
            
            # 计算验证集R²
            ss_res = np.sum((y_val.values - y_val_pred) ** 2)
            ss_tot = np.sum((y_val.values - np.mean(y_val.values)) ** 2)
            val_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            mse_scores.append(val_mse)
            r2_scores.append(val_r2)
    
    return np.mean(mse_scores), np.std(mse_scores), np.mean(r2_scores), np.std(r2_scores)

def evaluate_with_simple_models(datasets):
    """使用简单模型评估性能"""
    print("\n=== 使用指定特征预测MAD标签并计算MSE ===")
    
    results = {}
    
    for dist, data_dict in datasets.items():
        print(f"\n{dist} 距离分析:")
        
        X = data_dict['features']
        y = data_dict['target']
        
        print(f"  特征数量: {X.shape[1]}")
        print(f"  样本数量: {len(y)}")
        print(f"  MAD值范围: {y.min():.4f} ~ {y.max():.4f}")
        print(f"  MAD均值: {y.mean():.4f}")
        
        # 使用简单线性回归
        print(f"\n  使用简单线性回归:")
        
        try:
            mean_mse, std_mse, mean_r2, std_r2 = cross_validation_mse(X, y, k_folds=5)
            
            print(f"    交叉验证MSE: {mean_mse:.6f} (±{std_mse:.6f})")
            print(f"    交叉验证R²: {mean_r2:.4f} (±{std_r2:.4f})")
            
            results[dist] = {
                'mse_mean': mean_mse,
                'mse_std': std_mse,
                'r2_mean': mean_r2,
                'r2_std': std_r2
            }
            
        except Exception as e:
            print(f"    计算出错: {e}")
            results[dist] = {
                'mse_mean': np.inf,
                'mse_std': np.inf,
                'r2_mean': -np.inf,
                'r2_std': np.inf,
                'error': str(e)
            }
    
    return results

def create_summary_table(results):
    """创建汇总表"""
    print("\n=== 结果汇总表 ===")
    
    summary_data = []
    
    for dist in ['5mm', '10mm', '15mm', '20mm']:
        if dist in results:
            result = results[dist]
            summary_data.append({
                '距离': dist,
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
    with open('specific_features_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 指定特征预测MAD标签分析报告 ===\n\n")
        f.write("特征范围: feature_0 到 feature_9 和 feature_80 到 feature_97\n\n")
        
        for dist, result in results.items():
            f.write(f"{dist} 距离结果:\n")
            f.write(f"  MSE均值: {result['mse_mean']:.6f}\n")
            f.write(f"  MSE标准差: {result['mse_std']:.6f}\n")
            f.write(f"  R²均值: {result['r2_mean']:.4f}\n")
            f.write(f"  R²标准差: {result['r2_std']:.4f}\n")
            if 'error' in result:
                f.write(f"  错误: {result['error']}\n")
            f.write("\n")
    
    print("详细报告已保存到: specific_features_analysis_report.txt")

def main():
    """主函数"""
    print("=== 使用指定特征预测MAD标签 ===")
    print("特征范围: feature_0 到 feature_9 和 feature_80 到 feature_97")
    
    # 1. 加载数据
    datasets = load_mad_data()
    
    if not datasets:
        print("没有成功加载任何数据，程序退出")
        return
    
    # 2. 选择指定特征
    selected_datasets = select_specific_features(datasets)
    
    if not selected_datasets:
        print("没有找到指定特征，程序退出")
        return
    
    # 3. 评估模型性能
    results = evaluate_with_simple_models(selected_datasets)
    
    # 4. 创建汇总表
    summary_df = create_summary_table(results)
    
    # 5. 保存结果
    save_results(results, summary_df)
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()


