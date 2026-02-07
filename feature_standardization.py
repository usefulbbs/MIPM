import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def standardize_features_across_distances(datasets):
    """跨距离标准化特征，消除量纲差异"""
    print("正在执行跨距离特征标准化...")
    
    # 收集所有距离的数据来计算全局统计量
    all_features = []
    for dist, data in datasets.items():
        all_features.append(data['features'])
    
    # 计算全局的均值和标准差
    global_mean = pd.concat(all_features, axis=0).mean()
    global_std = pd.concat(all_features, axis=0).std()
    
    print(f"全局特征均值范围: {global_mean.min():.4f} ~ {global_mean.max():.4f}")
    print(f"全局特征标准差范围: {global_std.min():.4f} ~ {global_std.max():.4f}")
    
    standardized_datasets = {}
    
    for dist, data in datasets.items():
        X = data['features']
        y = data['target']
        
        # 使用全局统计量进行标准化
        X_standardized = (X - global_mean) / global_std
        
        standardized_datasets[dist] = {
            'data': pd.concat([data['data'].iloc[:, :2], X_standardized], axis=1),
            'features': X_standardized,
            'target': y,
            'feature_names': X.columns.tolist()
        }
        
        print(f"{dist}: 标准化后特征范围: {X_standardized.min().min():.4f} ~ {X_standardized.max().max():.4f}")
    
    return standardized_datasets

def standardize_target_across_distances(datasets):
    """标准化目标变量，消除不同距离下MAD值的差异"""
    print("正在执行目标变量标准化...")
    
    # 收集所有距离的MAD值
    all_mad_values = []
    for dist, data in datasets.items():
        all_mad_values.extend(data['target'].values)
    
    # 计算全局MAD的均值和标准差
    global_mad_mean = np.mean(all_mad_values)
    global_mad_std = np.std(all_mad_values)
    
    print(f"全局MAD均值: {global_mad_mean:.4f}")
    print(f"全局MAD标准差: {global_mad_std:.4f}")
    
    standardized_datasets = {}
    
    for dist, data in datasets.items():
        X = data['features']
        y = data['target']
        
        # 标准化MAD值
        y_standardized = (y - global_mad_mean) / global_mad_std
        
        standardized_datasets[dist] = {
            'data': data['data'].copy(),
            'features': X,
            'target': y_standardized,
            'feature_names': X.columns.tolist()
        }
        # 更新data中的MAD列
        standardized_datasets[dist]['data'].iloc[:, 1] = y_standardized
        
        print(f"{dist}: MAD标准化后范围: {y_standardized.min():.4f} ~ {y_standardized.max():.4f}")
    
    return standardized_datasets

def calculate_feature_importance(datasets):
    """计算每个距离下的特征重要性"""
    print("正在计算特征重要性...")
    
    importance_results = {}
    
    for dist, data in datasets.items():
        print(f"  计算 {dist} 距离的特征重要性...")
        
        X = data['features']
        y = data['target']
        
        # 使用随机森林计算特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 获取特征重要性
        importance_scores = pd.Series(rf.feature_importances_, index=X.columns)
        importance_scores = importance_scores.sort_values(ascending=False)
        
        importance_results[dist] = {
            'mdi_importance': importance_scores,
            'model': rf
        }
        
        print(f"    {dist}: 最高重要性特征 {importance_scores.index[0]}: {importance_scores.iloc[0]:.4f}")
    
    return importance_results

def normalize_importance_scores(importance_results):
    """归一化不同距离下的特征重要性分数"""
    print("正在归一化特征重要性分数...")
    
    normalized_results = {}
    
    for dist, result in importance_results.items():
        # 获取重要性分数
        importance_scores = result['mdi_importance']
        
        # 归一化到0-1范围
        min_score = importance_scores.min()
        max_score = importance_scores.max()
        
        if max_score > min_score:
            normalized_scores = (importance_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = importance_scores * 0  # 如果所有值相同，设为0
        
        # 创建新的结果
        normalized_result = result.copy()
        normalized_result['mdi_importance'] = normalized_scores
        
        normalized_results[dist] = normalized_result
        
        print(f"  {dist}: 归一化后范围: {normalized_scores.min():.4f} ~ {normalized_scores.max():.4f}")
    
    return normalized_results

def check_consistency_improvement(original_importance, normalized_importance):
    """检查标准化后的一致性改善情况"""
    print("=== 一致性改善检查 ===")
    
    # 计算每个特征在不同距离下的重要性一致性
    feature_consistency = {}
    
    for feature in original_importance['5mm']['mdi_importance'].index:
        # 原始重要性的一致性
        original_scores = []
        for dist in original_importance.keys():
            original_scores.append(original_importance[dist]['mdi_importance'][feature])
        
        original_cv = np.std(original_scores) / np.mean(original_scores) if np.mean(original_scores) > 0 else 0
        
        # 标准化后重要性的一致性
        normalized_scores = []
        for dist in normalized_importance.keys():
            normalized_scores.append(normalized_importance[dist]['mdi_importance'][feature])
        
        normalized_cv = np.std(normalized_scores) / np.mean(normalized_scores) if np.mean(normalized_scores) > 0 else 0
        
        # 改善程度
        if original_cv > 0:
            improvement = (original_cv - normalized_cv) / original_cv * 100
        else:
            improvement = 0
        
        feature_consistency[feature] = {
            'original_cv': original_cv,
            'normalized_cv': normalized_cv,
            'improvement_percent': improvement
        }
    
    # 排序显示改善最大的特征
    sorted_features = sorted(feature_consistency.items(), 
                           key=lambda x: x[1]['improvement_percent'], reverse=True)
    
    print(f"\n改善最大的前20个特征:")
    for i, (feature, info) in enumerate(sorted_features[:20]):
        print(f"{i+1:2d}. {feature}: {info['improvement_percent']:.1f}% 改善")
    
    return feature_consistency

def visualize_consistency_comparison(original_importance, normalized_importance, feature_consistency):
    """可视化一致性改善情况"""
    print("正在生成可视化图表...")
    
    # 选择改善最大的前10个特征
    top_improved = sorted(feature_consistency.items(), 
                         key=lambda x: x[1]['improvement_percent'], reverse=True)[:10]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('特征重要性一致性改善分析', fontsize=16)
    
    # 1. 改善程度排名
    ax1 = axes[0, 0]
    features = [f[0] for f in top_improved]
    improvements = [f[1]['improvement_percent'] for f in top_improved]
    
    bars = ax1.barh(range(len(features)), improvements)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels([f'F{i+1}' for i in range(len(features))])
    ax1.set_xlabel('改善程度 (%)')
    ax1.set_title('前10个改善最大的特征')
    ax1.grid(True, alpha=0.3)
    
    # 2. 原始CV vs 标准化后CV
    ax2 = axes[0, 1]
    original_cvs = [f[1]['original_cv'] for f in top_improved]
    normalized_cvs = [f[1]['normalized_cv'] for f in top_improved]
    
    x_pos = np.arange(len(features))
    width = 0.35
    
    ax2.bar(x_pos - width/2, original_cvs, width, label='标准化前', alpha=0.8)
    ax2.bar(x_pos + width/2, normalized_cvs, width, label='标准化后', alpha=0.8)
    
    ax2.set_xlabel('特征')
    ax2.set_ylabel('变异系数 (CV)')
    ax2.set_title('标准化前后CV对比')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'F{i+1}' for i in range(len(features))], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 前5个特征在不同距离下的原始重要性
    ax3 = axes[1, 0]
    top_5_features = [f[0] for f in top_improved[:5]]
    
    distances = list(original_importance.keys())
    x_pos = np.arange(len(distances))
    
    for i, feature in enumerate(top_5_features):
        scores = [original_importance[dist]['mdi_importance'][feature] for dist in distances]
        ax3.plot(x_pos, scores, marker='o', label=f'F{i+1}', linewidth=2, markersize=8)
    
    ax3.set_xlabel('距离')
    ax3.set_ylabel('原始重要性')
    ax3.set_title('前5个特征在不同距离下的原始重要性')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(distances)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 前5个特征在不同距离下的标准化后重要性
    ax4 = axes[1, 1]
    
    for i, feature in enumerate(top_5_features):
        scores = [normalized_importance[dist]['mdi_importance'][feature] for dist in distances]
        ax4.plot(x_pos, scores, marker='s', label=f'F{i+1}', linewidth=2, markersize=8)
    
    ax4.set_xlabel('距离')
    ax4.set_ylabel('标准化后重要性')
    ax4.set_title('前5个特征在不同距离下的标准化后重要性')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(distances)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('consistency_improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()



def save_mda_importance_to_csv(importance_results, data_type="MAD"):
    """将MDA特征重要性保存到CSV文件"""
    print(f"\n正在保存{data_type} MDA特征重要性到CSV文件...")
    
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为每个距离创建单独的CSV文件
    for dist, result in importance_results.items():
        # 获取MDA重要性分数
        importance_scores = result['mdi_importance']
        
        # 创建DataFrame，包含特征名、重要性分数和排名
        importance_df = pd.DataFrame({
            'feature_name': importance_scores.index,
            'mda_importance': importance_scores.values,
            'rank': range(1, len(importance_scores) + 1)
        })
        
        # 按重要性降序排列
        importance_df = importance_df.sort_values('mda_importance', ascending=False).reset_index(drop=True)
        
        # 重新计算排名
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # 保存到CSV文件
        filename = f'{data_type.lower()}_mda_importance_{dist}.csv'
        filepath = os.path.join(output_dir, filename)
        importance_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  {dist}: 已保存到 {filepath}")
        print(f"    前10个重要特征:")
        for i, row in importance_df.head(10).iterrows():
            print(f"      {row['rank']:2d}. {row['feature_name']}: {row['mda_importance']:.4f}")
    
    # 创建综合排名文件（所有距离的平均重要性）
    print(f"\n正在创建{data_type}综合MDA特征重要性排名...")
    
    # 收集所有特征
    all_features = set()
    for result in importance_results.values():
        all_features.update(result['mdi_importance'].index)
    
    # 计算每个特征的平均重要性
    data = []
    for feature in all_features:
        row = {'feature_name': feature}
        
        # 添加每个距离的重要性
        for dist, result in importance_results.items():
            if feature in result['mdi_importance']:
                row[f'{dist}_mda'] = result['mdi_importance'][feature]
            else:
                row[f'{dist}_mda'] = 0
        
        # 计算平均重要性
        mda_cols = [col for col in row.keys() if col.endswith('_mda')]
        row['avg_mda_importance'] = np.mean([row[col] for col in mda_cols])
        
        data.append(row)
    
    # 创建综合DataFrame
    comprehensive_df = pd.DataFrame(data)
    
    # 按平均重要性降序排列
    comprehensive_df = comprehensive_df.sort_values('avg_mda_importance', ascending=False).reset_index(drop=True)
    
    # 添加排名
    comprehensive_df['rank'] = range(1, len(comprehensive_df) + 1)
    
    # 重新排列列顺序
    cols = ['rank', 'feature_name', 'avg_mda_importance'] + [col for col in comprehensive_df.columns if col.endswith('_mda')]
    comprehensive_df = comprehensive_df[cols]
    
    # 保存综合排名
    filename = f"{data_type}_comprehensive_mda_ranking.csv"
    filepath = os.path.join(output_dir, filename)
    comprehensive_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"综合MDA特征重要性排名已保存到: {filepath}")
    
    # 显示前20个重要特征
    print(f"\n{data_type}前20个重要特征:")
    for _, row in comprehensive_df.head(20).iterrows():
        print(f"      {row['rank']:2d}. {row['feature_name']}: 平均MDA重要性={row['avg_mda_importance']:.4f}")
    
    print(f"\n所有{data_type} MDA特征重要性文件已保存完成！")
    return filepath

def save_original_importance_to_csv(importance_results, data_type="MAD"):
    """将原始的特征重要性保存到CSV文件（不归一化）"""
    print(f"\n正在保存原始{data_type}特征重要性到CSV文件...")
    
    # 为每个距离创建CSV文件
    for dist, result in importance_results.items():
        # 获取原始的重要性分数
        importance_scores = result['mdi_importance']
        
        # 创建DataFrame，包含特征名、重要性分数和排名
        importance_df = pd.DataFrame({
            'feature_name': importance_scores.index,
            'original_importance': importance_scores.values,
            'rank': range(1, len(importance_scores) + 1)
        })
        
        # 按重要性降序排列
        importance_df = importance_df.sort_values('original_importance', ascending=False).reset_index(drop=True)
        
        # 重新计算排名
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # 创建输出目录
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存到CSV文件
        filename = f'original_{data_type.lower()}_importance_{dist}.csv'
        filepath = os.path.join(output_dir, filename)
        importance_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  {dist}: 已保存到 {filepath}")
        print(f"    前5个特征:")
        for i, row in importance_df.head().iterrows():
            print(f"      {row['rank']:2d}. {row['feature_name']}: {row['original_importance']:.4f}")
    
    print(f"\n所有原始{data_type}特征重要性文件已保存完成！")

def main():
    """主函数 - 简化为只使用MDA特征选择"""
    print("=== 特征标准化MDA特征选择程序 ===")
    
    # 默认使用MAD数据
    print("\n=== MAD数据分析 ===")
    from run import load_and_preprocess_data
    datasets = load_and_preprocess_data()
    
    # 1. 计算原始MDA特征重要性
    print("\n=== 计算原始MDA特征重要性 ===")
    original_importance = calculate_feature_importance(datasets)
    
    # 2. 执行特征标准化
    print("\n=== 执行特征标准化 ===")
    datasets_feature_std = standardize_features_across_distances(datasets)
    datasets_target_std = standardize_target_across_distances(datasets_feature_std)
    
    # 3. 计算标准化后的MDA特征重要性
    print("\n=== 计算标准化后的MDA特征重要性 ===")
    new_importance = calculate_feature_importance(datasets_target_std)
    
    # 4. 特征重要性归一化
    print("\n=== 特征重要性归一化 ===")
    normalized_importance = normalize_importance_scores(new_importance)
    
    # 5. 检查一致性改善
    print("\n=== 检查一致性改善 ===")
    feature_consistency = check_consistency_improvement(original_importance, normalized_importance)
    
    # 6. 可视化结果
    print("\n=== 生成可视化结果 ===")
    visualize_consistency_comparison(original_importance, normalized_importance, feature_consistency)
    
    # 7. 保存MDA特征重要性到CSV文件
    print("\n=== 保存MDA特征重要性 ===")
    save_mda_importance_to_csv(original_importance, "MAD")
    save_mda_importance_to_csv(new_importance, "MAD_Standardized")
    
    print("\n=== 标准化完成 ===")
    print("现在不同距离下的MDA重要性波段应该更加一致了！")
    print("MDA特征重要性已保存到CSV文件中！")
    
    return datasets_target_std, original_importance, new_importance

def run_feature_standardization():
    """运行特征标准化的简单接口"""
    return main()

if __name__ == "__main__":
    main()