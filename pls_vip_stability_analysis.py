 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLS+VIP+稳定性选择分析不同距离数据集的关键特征

该模块结合了三种方法：
1. PLS (Partial Least Squares) - 偏最小二乘回归
2. VIP (Variable Importance in Projection) - 变量重要性投影
3. Stability Selection - 稳定性选择

用于识别在不同距离下都重要的稳定特征
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_vip_scores(X, y, n_components=None):
    """
    计算VIP (Variable Importance in Projection) 分数
    
    参数:
    X: 特征矩阵
    y: 目标变量
    n_components: PLS组件数量，如果为None则自动选择
    
    返回:
    vip_scores: 每个特征的VIP分数
    """
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # 自动选择最优组件数量
    if n_components is None:
        best_score = -np.inf
        best_n_components = 1
        
        for n_comp in range(1, min(X.shape[1] + 1, 11)):  # 最多10个组件
            try:
                pls = PLSRegression(n_components=n_comp)
                scores = cross_val_score(pls, X_scaled, y_scaled, cv=5, scoring='r2')
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_n_components = n_comp
            except:
                continue
        
        n_components = best_n_components
        print(f"自动选择最优PLS组件数量: {n_components}")
    
    # 拟合PLS模型
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y_scaled)
    
    # 计算VIP分数
    t = pls.x_scores_  # X的得分
    q = pls.y_loadings_  # Y的载荷
    w = pls.x_loadings_  # X的载荷
    
    # 确保维度正确 - sklearn的y_loadings_返回(1, n_components)
    # 我们需要将其转换为(n_components,)以便索引
    if q.shape[0] == 1:
        q = q.flatten()  # 从(1, n_components)转换为(n_components,)
    
    # 处理单组件情况
    if n_components == 1:
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
    else:
        # 多组件情况，确保是2D
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
    
    # 添加调试信息
    print(f"调试信息 - t形状: {t.shape}, q形状: {q.shape}, w形状: {w.shape}, n_components: {n_components}")
    
    # VIP公式: sqrt(sum((w^2 * q^2 * t^2) / sum(t^2)))
    vip_scores = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        numerator = 0
        denominator = 0
        
        for j in range(n_components):
            numerator += (w[i, j] ** 2) * (q[j] ** 2) * np.sum(t[:, j] ** 2)
            denominator += np.sum(t[:, j] ** 2)
        
        if denominator > 0:
            vip_scores[i] = np.sqrt(numerator / denominator)
        else:
            vip_scores[i] = 0
    
    return vip_scores, n_components

def stability_selection(X, y, n_iterations=100, subsample_ratio=0.8, threshold=0.6):
    """
    执行稳定性选择
    
    参数:
    X: 特征矩阵 (DataFrame或numpy数组)
    y: 目标变量
    n_iterations: 迭代次数
    subsample_ratio: 子样本比例
    threshold: 选择阈值
    
    返回:
    stability_scores: 每个特征的稳定性分数
    selected_features: 被选择的特征索引
    """
    # 确保X是numpy数组
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
    
    n_samples, n_features = X_array.shape
    subsample_size = int(n_samples * subsample_ratio)
    
    # 存储每次迭代中每个特征被选择的次数
    feature_selection_counts = np.zeros(n_features)
    
    for i in range(n_iterations):
        # 随机子采样
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_sub = X_array[indices]
        y_sub = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        try:
            # 使用随机森林进行特征选择
            rf = RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=-1)
            selector = SelectFromModel(rf, threshold='median')
            selector.fit(X_sub, y_sub)
            
            # 记录被选择的特征
            selected_features = selector.get_support()
            feature_selection_counts += selected_features
            
        except Exception as e:
            print(f"迭代 {i} 失败: {e}")
            continue
    
    # 计算稳定性分数
    stability_scores = feature_selection_counts / n_iterations
    
    # 选择稳定性分数超过阈值的特征
    selected_features = np.where(stability_scores >= threshold)[0]
    
    return stability_scores, selected_features

def analyze_distance_dataset(X, y, feature_names, distance_name):
    """
    分析单个距离数据集
    
    参数:
    X: 特征矩阵
    y: 目标变量
    feature_names: 特征名称列表
    distance_name: 距离名称
    
    返回:
    results: 包含各种分析结果的字典
    """
    print(f"\n=== 分析 {distance_name} 数据集 ===")
    
    # 1. 计算VIP分数
    print("计算VIP分数...")
    vip_scores, n_components = calculate_vip_scores(X, y)
    
    # 2. 执行稳定性选择
    print("执行稳定性选择...")
    stability_scores, selected_features = stability_selection(X, y)
    
    # 3. 计算PLS回归系数
    print("计算PLS回归系数...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y_scaled)
    pls_coefficients = np.abs(pls.coef_.flatten())
    
    # 4. 创建结果DataFrame
    results_df = pd.DataFrame({
        'feature_name': feature_names,
        'vip_score': vip_scores,
        'stability_score': stability_scores,
        'pls_coefficient': pls_coefficients,
        'distance': distance_name
    })
    
    # 5. 计算综合重要性分数
    # 归一化各个分数
    vip_norm = (vip_scores - vip_scores.min()) / (vip_scores.max() - vip_scores.min() + 1e-8)
    stability_norm = (stability_scores - stability_scores.min()) / (stability_scores.max() - stability_scores.min() + 1e-8)
    pls_norm = (pls_coefficients - pls_coefficients.min()) / (pls_coefficients.max() - pls_coefficients.min() + 1e-8)
    
    # 综合分数 (加权平均)
    composite_score = 0.4 * vip_norm + 0.4 * stability_norm + 0.2 * pls_norm
    results_df['composite_score'] = composite_score
    
    # 按综合分数排序
    results_df = results_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    # 6. 打印结果摘要
    print(f"\n{distance_name} 数据集分析结果:")
    print(f"PLS组件数量: {n_components}")
    print(f"稳定性选择阈值: 0.6")
    print(f"被选择的特征数量: {len(selected_features)}")
    
    print(f"\n前10个重要特征:")
    for i, row in results_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature_name']:20s} | "
              f"VIP: {row['vip_score']:6.3f} | "
              f"稳定性: {row['stability_score']:6.3f} | "
              f"PLS系数: {row['pls_coefficient']:6.3f} | "
              f"综合分数: {row['composite_score']:6.3f}")
    
    return {
        'results_df': results_df,
        'vip_scores': vip_scores,
        'stability_scores': stability_scores,
        'pls_coefficients': pls_coefficients,
        'selected_features': selected_features,
        'n_components': n_components
    }

def compare_across_distances(datasets):
    """
    比较不同距离下的特征重要性
    
    参数:
    datasets: 包含不同距离数据集的字典
    
    返回:
    comparison_results: 比较结果
    """
    print("\n=== 跨距离特征重要性比较 ===")
    
    all_results = {}
    common_features = None
    
    # 分析每个距离数据集
    for distance_name, dataset_info in datasets.items():
        X = dataset_info['features']
        y = dataset_info['target']
        feature_names = dataset_info['feature_names']
        
        results = analyze_distance_dataset(X, y, feature_names, distance_name)
        all_results[distance_name] = results
        
        # 找出共同特征
        if common_features is None:
            common_features = set(feature_names)
        else:
            common_features = common_features.intersection(set(feature_names))
    
    print(f"\n所有距离数据集中的共同特征数量: {len(common_features)}")
    
    # 创建跨距离比较DataFrame
    comparison_data = []
    
    for feature in common_features:
        feature_data = {'feature_name': feature}
        
        for distance_name, results in all_results.items():
            feature_row = results['results_df'][results['results_df']['feature_name'] == feature].iloc[0]
            feature_data[f'{distance_name}_composite_score'] = feature_row['composite_score']
            feature_data[f'{distance_name}_vip_score'] = feature_row['vip_score']
            feature_data[f'{distance_name}_stability_score'] = feature_row['stability_score']
        
        comparison_data.append(feature_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 计算跨距离一致性分数
    distance_columns = [col for col in comparison_df.columns if col.endswith('_composite_score')]
    comparison_df['cross_distance_consistency'] = comparison_df[distance_columns].std(axis=1)
    comparison_df['mean_composite_score'] = comparison_df[distance_columns].mean(axis=1)
    
    # 按一致性排序（标准差越小越一致）
    comparison_df = comparison_df.sort_values('cross_distance_consistency').reset_index(drop=True)
    
    return {
        'all_results': all_results,
        'comparison_df': comparison_df,
        'common_features': common_features
    }

def visualize_results(comparison_results, data_type="MAD"):
    """
    可视化分析结果
    
    参数:
    comparison_results: 比较结果字典
    data_type: 数据类型 ("MAD" 或 "VAD")
    """
    print(f"\n=== 生成{data_type}数据可视化结果 ===")
    
    all_results = comparison_results['all_results']
    comparison_df = comparison_results['comparison_df']
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{data_type}数据 - PLS+VIP+稳定性选择分析结果', fontsize=16, fontweight='bold')
    
    # 1. 跨距离一致性热图
    distance_names = list(all_results.keys())
    consistency_matrix = []
    
    for feature in comparison_df['feature_name'].head(20):  # 前20个特征
        row = []
        for distance in distance_names:
            feature_row = all_results[distance]['results_df'][
                all_results[distance]['results_df']['feature_name'] == feature
            ].iloc[0]
            row.append(feature_row['composite_score'])
        consistency_matrix.append(row)
    
    consistency_matrix = np.array(consistency_matrix)
    
    im1 = axes[0, 0].imshow(consistency_matrix, cmap='RdYlBu_r', aspect='auto')
    axes[0, 0].set_title('跨距离特征重要性一致性热图')
    axes[0, 0].set_xlabel('距离')
    axes[0, 0].set_ylabel('特征')
    axes[0, 0].set_xticks(range(len(distance_names)))
    axes[0, 0].set_xticklabels(distance_names, rotation=45)
    axes[0, 0].set_yticks(range(min(20, len(comparison_df))))
    axes[0, 0].set_yticklabels(comparison_df['feature_name'].head(20), fontsize=8)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 前20个特征的一致性分数
    top_features = comparison_df.head(20)
    axes[0, 1].barh(range(len(top_features)), top_features['cross_distance_consistency'])
    axes[0, 1].set_title('前20个特征的一致性分数')
    axes[0, 1].set_xlabel('一致性分数 (标准差)')
    axes[0, 1].set_ylabel('特征')
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature_name'], fontsize=8)
    axes[0, 1].invert_yaxis()
    
    # 3. VIP分数分布
    vip_scores_all = []
    distance_labels = []
    
    for distance_name, results in all_results.items():
        vip_scores_all.extend(results['vip_scores'])
        distance_labels.extend([distance_name] * len(results['vip_scores']))
    
    vip_df = pd.DataFrame({
        'vip_score': vip_scores_all,
        'distance': distance_labels
    })
    
    sns.boxplot(data=vip_df, x='distance', y='vip_score', ax=axes[1, 0])
    axes[1, 0].set_title('不同距离下的VIP分数分布')
    axes[1, 0].set_xlabel('距离')
    axes[1, 0].set_ylabel('VIP分数')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 稳定性分数分布
    stability_scores_all = []
    distance_labels = []
    
    for distance_name, results in all_results.items():
        stability_scores_all.extend(results['stability_scores'])
        distance_labels.extend([distance_name] * len(results['stability_scores']))
    
    stability_df = pd.DataFrame({
        'stability_score': stability_scores_all,
        'distance': distance_labels
    })
    
    sns.boxplot(data=stability_df, x='distance', y='stability_score', ax=axes[1, 1])
    axes[1, 1].set_title('不同距离下的稳定性分数分布')
    axes[1, 1].set_xlabel('距离')
    axes[1, 1].set_ylabel('稳定性分数')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = "output"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, f'{data_type.lower()}_pls_vip_stability_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"{data_type}数据可视化结果已保存到: {os.path.join(output_dir, f'{data_type.lower()}_pls_vip_stability_analysis.png')}")
    
    plt.show()

def save_results_to_csv(comparison_results, data_type="MAD"):
    """
    保存分析结果到CSV文件
    
    参数:
    comparison_results: 比较结果字典
    data_type: 数据类型 ("MAD" 或 "VAD")
    """
    print(f"\n=== 保存{data_type}数据分析结果到CSV文件 ===")
    
    all_results = comparison_results['all_results']
    comparison_df = comparison_results['comparison_df']
    
    # 创建输出目录
    output_dir = "output"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 保存跨距离比较结果
    comparison_file = os.path.join(output_dir, f'{data_type.lower()}_cross_distance_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f"{data_type}数据跨距离比较结果已保存到: {comparison_file}")
    
    # 2. 保存每个距离的详细结果
    for distance_name, results in all_results.items():
        # 保存综合结果
        composite_file = os.path.join(output_dir, f'{data_type.lower()}_{distance_name}_composite_analysis.csv')
        results['results_df'].to_csv(composite_file, index=False, encoding='utf-8-sig')
        print(f"{data_type}数据 {distance_name} 综合分析结果已保存到: {composite_file}")
        
        # 保存稳定性选择结果
        stability_file = os.path.join(output_dir, f'{data_type.lower()}_{distance_name}_stability_selection.csv')
        stability_df = pd.DataFrame({
            'feature_name': results['results_df']['feature_name'],
            'stability_score': results['stability_scores'],
            'selected': [i in results['selected_features'] for i in range(len(results['results_df']))]
        })
        stability_df.to_csv(stability_file, index=False, encoding='utf-8-sig')
        print(f"{data_type}数据 {distance_name} 稳定性选择结果已保存到: {stability_file}")
    
    # 3. 保存前50个最一致的特征
    top_consistent_file = os.path.join(output_dir, f'{data_type.lower()}_top_50_consistent_features.csv')
    top_50 = comparison_df.head(50)[['feature_name', 'cross_distance_consistency', 'mean_composite_score']]
    top_50.to_csv(top_consistent_file, index=False, encoding='utf-8-sig')
    print(f"{data_type}数据前50个最一致特征已保存到: {top_consistent_file}")
    
    print(f"\n{data_type}数据所有结果文件保存完成！")

def main():
    """
    主函数
    """
    print("=== PLS+VIP+稳定性选择分析程序 ===")
    
    # 导入数据预处理模块
    try:
        from run import load_and_preprocess_data, load_and_preprocess_vad_data
        print("正在加载MAD数据...")
        mad_datasets = load_and_preprocess_data()
        print("✓ MAD数据加载成功")
        
        print("\n正在加载VAD数据...")
        vad_datasets = load_and_preprocess_vad_data()
        print("✓ VAD数据加载成功")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 执行MAD数据分析
    print("\n" + "="*50)
    print("开始执行MAD数据PLS+VIP+稳定性选择分析...")
    print("="*50)
    
    # 1. MAD数据跨距离比较分析
    mad_comparison_results = compare_across_distances(mad_datasets)
    
    # 2. MAD数据可视化结果
    visualize_results(mad_comparison_results, "MAD")
    
    # 3. MAD数据保存结果
    save_results_to_csv(mad_comparison_results, "MAD")
    
    # 执行VAD数据分析
    print("\n" + "="*50)
    print("开始执行VAD数据PLS+VIP+稳定性选择分析...")
    print("="*50)
    
    # 1. VAD数据跨距离比较分析
    vad_comparison_results = compare_across_distances(vad_datasets)
    
    # 2. VAD数据可视化结果
    visualize_results(vad_comparison_results, "VAD")
    
    # 3. VAD数据保存结果
    save_results_to_csv(vad_comparison_results, "VAD")
    
    print("\n=== 分析完成 ===")
    print("MAD和VAD数据的PLS+VIP+稳定性选择分析已完成！")
    print("关键特征已识别并保存到CSV文件中！")
    
    return {
        'mad_results': mad_comparison_results,
        'vad_results': vad_comparison_results
    }

if __name__ == "__main__":
    main()