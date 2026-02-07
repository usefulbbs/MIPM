import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_mad_data():
    """加载MAD数据"""
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

def analyze_importance_consistency(datasets):
    """分析特征重要性的一致性"""
    print("\n=== 分析特征重要性一致性 ===")
    
    importance_results = {}
    
    for dist, data_dict in datasets.items():
        print(f"\n分析 {dist} 距离...")
        
        X = data_dict['features']
        y = data_dict['target']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # 获取特征重要性
        importance = rf.feature_importances_
        
        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_results[dist] = importance_df
        
        print(f"  前5个最重要特征:")
        for i, row in importance_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return importance_results

def cross_distance_consistency_analysis(importance_results):
    """跨距离一致性分析"""
    print("\n=== 跨距离一致性分析 ===")
    
    # 收集所有距离下的前50个重要特征
    all_top_features = {}
    for dist, result in importance_results.items():
        top_50 = result.head(50)[['feature', 'importance']].copy()
        top_50.columns = ['feature', f'{dist}_importance']
        all_top_features[dist] = top_50
    
    # 合并所有距离的重要性数据
    merged_importance = all_top_features['5mm'].copy()
    for dist in ['10mm', '15mm', '20mm']:
        merged_importance = merged_importance.merge(
            all_top_features[dist][['feature', f'{dist}_importance']], 
            on='feature', how='outer'
        )
    
    # 填充缺失值
    merged_importance = merged_importance.fillna(0)
    
    # 计算一致性指标
    distance_columns = ['5mm_importance', '10mm_importance', '15mm_importance', '20mm_importance']
    
    merged_importance['mean_importance'] = merged_importance[distance_columns].mean(axis=1)
    merged_importance['std_importance'] = merged_importance[distance_columns].std(axis=1)
    merged_importance['cv_importance'] = merged_importance['std_importance'] / (merged_importance['mean_importance'] + 1e-8)
    merged_importance['stability_score'] = 1 / (1 + merged_importance['cv_importance'])
    
    # 识别稳定的重要特征
    stable_threshold = merged_importance['stability_score'].quantile(0.8)  # 稳定性前20%
    importance_threshold = merged_importance['mean_importance'].quantile(0.8)  # 重要性前20%
    
    stable_features = merged_importance[
        (merged_importance['stability_score'] >= stable_threshold) &
        (merged_importance['mean_importance'] >= importance_threshold)
    ].copy()
    
    stable_features = stable_features.sort_values('mean_importance', ascending=False)
    
    print(f"\n一致性分析结果:")
    print(f"  总特征数: {len(merged_importance)}")
    print(f"  稳定重要特征数: {len(stable_features)}")
    print(f"  稳定性阈值: {stable_threshold:.4f}")
    print(f"  重要性阈值: {importance_threshold:.4f}")
    
    print(f"\n前15个最稳定的重要特征:")
    for i, row in stable_features.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: 平均重要性={row['mean_importance']:.4f}, 稳定性={row['stability_score']:.4f}")
        print(f"      各距离重要性: 5mm={row['5mm_importance']:.4f}, 10mm={row['10mm_importance']:.4f}, 15mm={row['15mm_importance']:.4f}, 20mm={row['20mm_importance']:.4f}")
    
    return stable_features, merged_importance

def identify_physical_bands(stable_features):
    """识别物理上合理的波段"""
    print("\n=== 物理波段分析 ===")
    
    # 提取波段数值
    band_values = []
    for feature in stable_features['feature']:
        try:
            if isinstance(feature, (int, float)):
                band_values.append(float(feature))
            elif isinstance(feature, str):
                import re
                numbers = re.findall(r'\d+\.?\d*', feature)
                if numbers:
                    band_values.append(float(numbers[0]))
        except:
            continue
    
    if band_values:
        band_values = sorted(band_values)
        print(f"  波段范围: {min(band_values):.1f} - {max(band_values):.1f} nm")
        
        # 分析波段分布
        print(f"  波段分布:")
        ranges = [(900, 1000), (1000, 1100), (1100, 1200), (1200, 1300), (1300, 1400), (1400, 1500)]
        for start, end in ranges:
            count = sum(1 for b in band_values if start <= b < end)
            if count > 0:
                print(f"    {start}-{end} nm: {count} 个波段")
    
    return band_values

def visualize_consistency(merged_importance, stable_features):
    """可视化一致性结果"""
    print("\n正在生成可视化图表...")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('跨距离特征重要性一致性分析', fontsize=16)
    
    # 1. 稳定性vs重要性散点图
    ax1 = axes[0, 0]
    ax1.scatter(merged_importance['mean_importance'], merged_importance['stability_score'], alpha=0.6)
    ax1.scatter(stable_features['mean_importance'], stable_features['stability_score'], 
                color='red', s=100, alpha=0.8, label='稳定重要特征')
    ax1.set_xlabel('平均重要性')
    ax1.set_ylabel('稳定性得分')
    ax1.set_title('特征重要性 vs 稳定性')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 各距离重要性对比（前20个特征）
    ax2 = axes[0, 1]
    top_20 = stable_features.head(20)
    x_pos = np.arange(len(top_20))
    
    width = 0.2
    ax2.bar(x_pos - width*1.5, top_20['5mm_importance'], width, label='5mm', alpha=0.8)
    ax2.bar(x_pos - width*0.5, top_20['10mm_importance'], width, label='10mm', alpha=0.8)
    ax2.bar(x_pos + width*0.5, top_20['15mm_importance'], width, label='15mm', alpha=0.8)
    ax2.bar(x_pos + width*1.5, top_20['20mm_importance'], width, label='20mm', alpha=0.8)
    
    ax2.set_xlabel('特征排名')
    ax2.set_ylabel('重要性')
    ax2.set_title('前20个稳定特征在各距离下的重要性')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'F{i+1}' for i in range(20)], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 稳定性分布直方图
    ax3 = axes[1, 0]
    ax3.hist(merged_importance['stability_score'], bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(stable_features['stability_score'].min(), color='red', linestyle='--', 
                label=f'稳定特征阈值: {stable_features["stability_score"].min():.3f}')
    ax3.set_xlabel('稳定性得分')
    ax3.set_ylabel('特征数量')
    ax3.set_title('特征稳定性分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 重要性分布直方图
    ax4 = axes[1, 1]
    ax4.hist(merged_importance['mean_importance'], bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(stable_features['mean_importance'].min(), color='red', linestyle='--',
                label=f'重要特征阈值: {stable_features["mean_importance"].min():.3f}')
    ax4.set_xlabel('平均重要性')
    ax4.set_ylabel('特征数量')
    ax4.set_title('特征重要性分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('band_consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== 解决波段重要性不一致问题 ===")
    
    # 1. 加载数据
    datasets = load_mad_data()
    
    # 2. 分析特征重要性
    importance_results = analyze_importance_consistency(datasets)
    
    # 3. 跨距离一致性分析
    stable_features, merged_importance = cross_distance_consistency_analysis(importance_results)
    
    # 4. 物理波段分析
    band_values = identify_physical_bands(stable_features)
    
    # 5. 可视化结果
    # visualize_consistency(merged_importance, stable_features)
    
    # 6. 保存结果
    merged_importance.to_csv('cross_distance_importance_analysis.csv', index=False, encoding='utf-8-sig')
    stable_features.to_csv('stable_important_features.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== 分析完成 ===")
    print(f"完整分析结果已保存到: cross_distance_importance_analysis.csv")
    print(f"稳定重要特征已保存到: stable_important_features.csv")
    print(f"可视化图表已保存到: band_consistency_analysis.png")
    
    # 7. 给出建议
    print(f"\n=== 建议 ===")
    print("1. 选择稳定性得分 > 0.8 的特征")
    print("2. 优先选择在多个距离下都表现良好的特征")
    print("3. 重点关注与灰分化学性质相关的波段范围")
    print("4. 考虑特征之间的相关性，避免冗余")

if __name__ == "__main__":
    main()