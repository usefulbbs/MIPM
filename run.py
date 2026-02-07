import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from correlation_r import correlation_analysis
from scipy import stats
from correlation_x import feature_importance_analysis, compare_correlation_and_importance
from mse_comparison import main as run_mse_comparison
from comprehensive_methods_comparison import main as run_comprehensive_comparison
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """加载并预处理四个距离的MAD数据文件"""
    print("正在加载MAD数据...")
    
    # 加载四个距离的数据
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
        
        # 添加调试信息
        print(f"\n{dist} 数据检查:")
        print(f"  特征数量: {X.shape[1]}")
        print(f"  样本数量: {len(y)}")
        print(f"  Mad值范围: {y.min():.4f} ~ {y.max():.4f}")
        print(f"  Mad均值: {y.mean():.4f}")
        print(f"  前5个特征的前5个值:")
        for i in range(min(5, X.shape[1])):
            feature_name = X.columns[i]
            values = X.iloc[:5, i].values
            print(f"    {feature_name}: {values}")
        
        # 检查是否有重复值或异常值
        print(f"  检查前10个特征是否有重复值:")
        for i in range(min(10, X.shape[1])):
            feature_name = X.columns[i]
            unique_count = X.iloc[:, i].nunique()
            print(f"    {feature_name}: {unique_count} 个唯一值")
    
    return datasets

def load_and_preprocess_vad_data():
    """加载并预处理四个距离的VAD数据文件"""
    print("正在加载VAD数据...")
    
    # 加载四个距离的数据
    distances = ['5mm', '10mm', '15mm', '20mm']
    datasets = {}
    
    for dist in distances:
        file_path = f'dataset_vad/vad_{dist}.xlsx'
        df = pd.read_excel(file_path)
        
        # 重命名列
        df.columns = ['sample_name', 'Vad'] + [f'feature_{i}' for i in range(len(df.columns)-2)]
        
        # 分离特征和目标变量
        X = df.iloc[:, 2:]  # 特征列
        y = df['Vad']        # 目标变量
        
        datasets[dist] = {
            'data': df,
            'features': X,
            'target': y,
            'feature_names': X.columns.tolist()
        }
        
        # 添加调试信息
        print(f"\n{dist} VAD数据检查:")
        print(f"  特征数量: {X.shape[1]}")
        print(f"  样本数量: {len(y)}")
        print(f"  Vad值范围: {y.min():.4f} ~ {y.max():.4f}")
        print(f"  Vad均值: {y.mean():.4f}")
        print(f"  前5个特征的前5个值:")
        for i in range(min(5, X.shape[1])):
            feature_name = X.columns[i]
            values = X.iloc[:5, i].values
            print(f"    {feature_name}: {values}")
        
        # 检查是否有重复值或异常值
        print(f"  检查前10个特征是否有重复值:")
        for i in range(min(10, X.shape[1])):
            feature_name = X.columns[i]
            unique_count = X.iloc[:, i].nunique()
            print(f"    {feature_name}: {unique_count} 个唯一值")
    
    return datasets


def visualize_correlations(datasets, correlation_results):
    """可视化相关性结果"""
    print("\n正在生成可视化图表...")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('不同距离下特征与Mad的相关性分析', fontsize=16)
    
    distances = list(datasets.keys())
    
    for i, dist in enumerate(distances):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # 获取前20个最相关的特征（使用距离相关性）
        top_features = correlation_results[dist]['distance_correlation'].head(20)
        
        # 绘制相关性条形图
        bars = ax.barh(range(len(top_features)), top_features['distance_correlation'])
        
        # 为负相关和正相关设置不同颜色
        for j, (bar, corr) in enumerate(zip(bars, top_features['distance_correlation'])):
            if corr < 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        ax.set_title(f'{dist} 距离 - 前20个最相关特征')
        ax.set_xlabel('距离相关系数')
        ax.set_ylabel('特征')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加特征标签
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f'F{i}' for i in range(1, 21)])
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def cross_distance_analysis(datasets, correlation_results):
    """跨距离的特征相关性一致性分析"""
    print("\n正在进行跨距离分析...")
    
    # 为每个距离选择前10个最相关的特征（使用距离相关性）
    top_features_by_distance = {}
    for dist in datasets.keys():
        top_features = correlation_results[dist]['distance_correlation'].head(10)['feature'].tolist()
        top_features_by_distance[dist] = top_features
    
    # 分析特征在不同距离下的一致性
    print("\n各距离下前10个最相关特征的一致性分析:")
    for dist, features in top_features_by_distance.items():
        print(f"\n{dist}:")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
    
    # 寻找在所有距离下都表现良好的特征
    all_features = set()
    for features in top_features_by_distance.values():
        all_features.update(features)
    
    feature_consistency = {}
    for feature in all_features:
        consistency_score = 0
        for dist in datasets.keys():
            if feature in top_features_by_distance[dist]:
                # 根据排名给分，排名越靠前分数越高
                rank = top_features_by_distance[dist].index(feature)
                consistency_score += (10 - rank)
        feature_consistency[feature] = consistency_score
    
    # 按一致性得分排序
    consistency_df = pd.DataFrame([
        {'feature': k, 'consistency_score': v}
        for k, v in feature_consistency.items()
    ]).sort_values('consistency_score', ascending=False)
    
    print(f"\n特征一致性排名 (前20个):")
    print(consistency_df.head(20))
    
    return consistency_df

def feature_selection_recommendation(correlation_results, consistency_df):
    """基于分析结果给出特征选择建议"""
    print("\n=== 特征选择建议 ===")
    
    print("\n1. 基于Kendall's Tau相关系数的建议:")
    for dist in correlation_results.keys():
        top_kendall = correlation_results[dist]['kendall'].head(5)
        print(f"\n{dist} 距离下最相关的5个特征:")
        for i, row in top_kendall.iterrows():
            print(f"  {row['feature']}: τ={row['correlation']:.4f}, p={row['p_value']:.4e}")
    
    print("\n2. 基于距离相关性的建议:")
    for dist in correlation_results.keys():
        top_dcor = correlation_results[dist]['distance_correlation'].head(5)
        print(f"\n{dist} 距离下距离相关性最高的5个特征:")
        for i, row in top_dcor.iterrows():
            print(f"  {row['feature']}: dCor={row['distance_correlation']:.4f}")
    
    print("\n3. 基于MIC的建议:")
    for dist in correlation_results.keys():
        top_mic = correlation_results[dist]['mic'].head(5)
        print(f"\n{dist} 距离下MIC最高的5个特征:")
        for i, row in top_mic.iterrows():
            print(f"  {row['feature']}: MIC={row['mic']:.4f}")
    
    print("\n4. 基于互信息的建议:")
    for dist in correlation_results.keys():
        top_mi = correlation_results[dist]['mutual_info'].head(5)
        print(f"\n{dist} 距离下互信息最高的5个特征:")
        for i, row in top_mi.iterrows():
            print(f"  {row['feature']}: MI={row['mutual_info']:.4f}")
    
    print("\n5. 跨距离一致性建议:")
    print("以下特征在多个距离下都表现良好:")
    top_consistent = consistency_df.head(10)
    for i, row in top_consistent.iterrows():
        print(f"  {row['feature']}: 一致性得分={row['consistency_score']}")
    
    print("\n6. 最终推荐:")
    print("建议选择以下特征进行建模:")
    recommended_features = consistency_df.head(15)['feature'].tolist()
    for i, feature in enumerate(recommended_features, 1):
        print(f"  {i:2d}. {feature}")

def main():
    """主函数"""
    print("=== 煤炭光谱特征分析程序 ===")
    
    # 选择要分析的数据类型
    data_type = input("请选择要分析的数据类型 (1: MAD数据, 2: VAD数据, 3: 两者都分析): ").strip()
    
    if data_type == "1" or data_type == "3":
        print("\n=== MAD数据分析 ===")
        # 1. 加载和预处理MAD数据
        mad_datasets = load_and_preprocess_data()
        
        # 2. 相关性分析
        # correlation_results = correlation_analysis(mad_datasets)

        # 3. 特征重要性分析（包含MDI和MDA）
        print("\n=== 特征重要性分析 ===")
        print("MDI (Mean Decrease in Impurity): sklearn默认方法，速度快")
        print("MDA (Mean Decrease in Accuracy): 打乱法，更稳健但计算慢")
        importance_results = feature_importance_analysis(mad_datasets)
        
        # 4. 跨距离波段一致性分析
        print("\n=== 跨距离波段一致性分析 ===")
        print("分析不同距离下波段重要性的一致性，识别稳定的重要波段")
        from correlation_x import analyze_band_consistency_across_distances, identify_physical_band_ranges, recommend_band_selection_strategy
        
        stable_bands = analyze_band_consistency_across_distances(mad_datasets, importance_results)
        band_values = identify_physical_band_ranges(stable_bands)
        recommend_band_selection_strategy(stable_bands, band_values)  

        # 4. MSE比较分析
        # print("\n=== MSE比较分析 ===")
        # print("比较不同特征选择方法的预测效果")
        
        # run_mse_comparison('mad')
        
        # # 5. 综合方法比较分析
        # print("\n=== MAD数据综合方法比较分析 ===")
        # run_comprehensive_comparison()
    
    if data_type == "2" or data_type == "3":
        print("\n=== VAD数据分析 ===")
        # 1. 加载和预处理VAD数据
        vad_datasets = load_and_preprocess_vad_data()
        
        # # 2. 相关性分析
        # correlation_results = correlation_analysis(vad_datasets)

        # 3. 特征重要性分析（包含MDI和MDA）
        print("\n=== 特征重要性分析 ===")
        print("MDI (Mean Decrease in Impurity): sklearn默认方法，速度快")
        print("MDA (Mean Decrease in Accuracy): 打乱法，更稳健但计算慢")
        importance_results = feature_importance_analysis(vad_datasets)
        
        # 4. 跨距离波段一致性分析
        print("\n=== 跨距离波段一致性分析 ===")
        print("分析不同距离下波段重要性的一致性，识别稳定的重要波段")
        from correlation_x import analyze_band_consistency_across_distances, identify_physical_band_ranges, recommend_band_selection_strategy
        
        stable_bands = analyze_band_consistency_across_distances(vad_datasets, importance_results)
        band_values = identify_physical_band_ranges(stable_bands)
        recommend_band_selection_strategy(stable_bands, band_values) 

        # 4. MSE比较分析
        # print("\n=== MSE比较分析 ===")
        # print("比较不同特征选择方法的预测效果")
        
        # run_mse_comparison('vad')
        
        # # 5. 综合方法比较分析2
        # # print("\n=== VAD数据综合方法比较分析 ===")
        # # # 调用VAD数据的综合方法比较分析
        # run_comprehensive_comparison('vad')
if __name__ == "__main__":
    main()