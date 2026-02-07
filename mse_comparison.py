import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def load_feature_rankings(data_type='mad'):
    """加载特征排序结果"""
    print(f"正在加载{data_type.upper()}数据的特征排序结果...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    feature_rankings = {}
    
    # 加载相关性分析的特征排序
    correlation_rankings = {}
    for dist in distances:
        if data_type.lower() == 'mad':
            file_path = f'results/correlation/feature_ranking_{dist}.csv'
        else:  # vad
            file_path = f'results_vad/correlation/feature_ranking_{dist}.csv'
        df = pd.read_csv(file_path)
        correlation_rankings[dist] = df
    
    # 加载模型分析的特征排序
    model_rankings = {}
    for dist in distances:
        if data_type.lower() == 'mad':
            file_path = f'results/model/feature_importance_{dist}.csv'
        else:  # vad
            file_path = f'results_vad/model/feature_importance_{dist}.csv'
        df = pd.read_csv(file_path)
        model_rankings[dist] = df
    
    feature_rankings['correlation'] = correlation_rankings
    feature_rankings['model'] = model_rankings
    
    return feature_rankings

def load_original_data(data_type='mad'):
    """加载原始数据"""
    print(f"正在加载{data_type.upper()}数据...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    datasets = {}
    
    for dist in distances:
        if data_type.lower() == 'mad':
            file_path = f'dataset_mad/mad_{dist}.xlsx'
            target_col = 'Mad'
        else:  # vad
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
    
    return datasets

def select_features_by_ranking(feature_ranking, n_features, method='top'):
    """根据特征排序选择特征"""
    if method == 'top':
        # 选择排名前n_features的特征
        selected_features = feature_ranking.head(n_features)['feature'].tolist()
    elif method in ['kendall_rank', 'spearman_rank', 'dcor_rank', 'mic_rank', 'mi_rank']:
        # 相关性分析方法：按对应排名列排序
        sorted_features = feature_ranking.sort_values(method).head(n_features)
        selected_features = sorted_features['feature'].tolist()
    elif method in ['rf_importance', 'mda_score', 'xgb_importance', 'lr_coefficient']:
        # 模型重要性方法：按对应重要性列排序（降序）
        sorted_features = feature_ranking.sort_values(method, ascending=False).head(n_features)
        selected_features = sorted_features['feature'].tolist()
    else:
        # 默认方法
        selected_features = feature_ranking.head(n_features)['feature'].tolist()
    
    return selected_features

def evaluate_model_performance(X, y, feature_subset, model_type='rf', cv_folds=5):
    """评估模型性能"""
    if len(feature_subset) == 0:
        return np.inf, np.inf
    
    # 选择特征子集
    X_subset = X[feature_subset]
    
    # 创建模型
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LinearRegression()
    
    # 交叉验证
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_idx, val_idx in kf.split(X_subset):
        X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    return mean_mse, std_mse

def compare_feature_selection_methods(datasets, feature_rankings):
    """比较9种不同特征选择方法的MSE"""
    print("正在比较9种不同特征选择方法的MSE...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    feature_counts = [10, 20, 30, 50]  # 测试不同特征数量
    model_types = ['rf', 'xgb', 'lr']   # 测试不同模型
    
    # 定义9种特征选择方法
    feature_selection_methods = [
        'kendall_rank',      # Kendall's Tau排名
        'spearman_rank',     # Spearman排名
        'dcor_rank',         # 距离相关性排名
        'mic_rank',          # MIC排名
        'mi_rank',           # 互信息排名
        'rf_importance',     # 随机森林重要性
        'mda_score',         # MDA得分
        'xgb_importance',    # XGBoost重要性
        'lr_coefficient'     # 线性回归系数
    ]
    
    results = {}
    
    for dist in distances:
        print(f"\n分析 {dist} 距离...")
        X = datasets[dist]['features']
        y = datasets[dist]['target']
        
        dist_results = {}
        
        # 对每种特征选择方法进行分析
        for method in feature_selection_methods:
            print(f"  {method} 特征选择...")
            method_mse = {}
            
            for n_features in feature_counts:
                # 根据不同的方法选择特征
                if method in ['kendall_rank', 'spearman_rank', 'dcor_rank', 'mic_rank', 'mi_rank']:
                    # 相关性分析方法
                    selected_features = select_features_by_ranking(
                        feature_rankings['correlation'][dist], 
                        n_features, 
                        method
                    )
                else:
                    # 模型重要性方法
                    selected_features = select_features_by_ranking(
                        feature_rankings['model'][dist], 
                        n_features, 
                        method
                    )
                
                for model_type in model_types:
                    mse, std = evaluate_model_performance(X, y, selected_features, model_type)
                    key = f'{method}_{n_features}_{model_type}'
                    method_mse[key] = {'mse': mse, 'std': std, 'n_features': n_features}
            
            dist_results[method] = method_mse
        
        results[dist] = dist_results
    
    return results

def create_mse_comparison_table(results, data_type='mad'):
    """创建MSE比较表"""
    print("正在创建MSE比较表...")
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    methods = ['kendall_rank', 'spearman_rank', 'dcor_rank', 'mic_rank', 'mi_rank',
               'rf_importance', 'mda_score', 'xgb_importance', 'lr_coefficient']
    feature_counts = [10, 20, 30, 50]
    model_types = ['rf', 'xgb', 'lr']
    
    # 创建结果DataFrame
    comparison_data = []
    
    for dist in distances:
        for method in methods:
            for n_features in feature_counts:
                for model_type in model_types:
                    key = f'{method}_{n_features}_{model_type}'
                    
                    if key in results[dist][method]:
                        mse_info = results[dist][method][key]
                        comparison_data.append({
                            'distance': dist,
                            'method': method,
                            'n_features': mse_info['n_features'],
                            'model_type': model_type,
                            'mse': mse_info['mse'],
                            'std': mse_info['std']
                        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存结果
    if data_type.lower() == 'mad':
        comparison_df.to_csv('results/mse_comparison_results.csv', index=False)
    else:
        comparison_df.to_csv('results_vad/mse_comparison_results.csv', index=False)
    
    return comparison_df

def visualize_mse_comparison(comparison_df, data_type='mad'):
    """可视化MSE比较结果"""
    print("正在生成可视化图表...")
    
    # 设置中文字体
    if plt is not None:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 不同方法的MSE比较（按距离）
    if plt is None:
        print("警告: matplotlib不可用，跳过图表生成")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{data_type.upper()}数据不同特征选择方法的MSE比较', fontsize=16)
    
    distances = ['5mm', '10mm', '15mm', '20mm']
    
    for i, dist in enumerate(distances):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        dist_data = comparison_df[comparison_df['distance'] == dist]
        
        # 按方法分组（9种方法）
        methods = ['kendall_rank', 'spearman_rank', 'dcor_rank', 'mic_rank', 'mi_rank',
                   'rf_importance', 'mda_score', 'xgb_importance', 'lr_coefficient']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        
        for j, method in enumerate(methods):
            method_data = dist_data[dist_data['method'] == method]
            if not method_data.empty:
                # 按特征数量分组
                feature_counts = sorted(method_data['n_features'].unique())
                mse_means = []
                mse_stds = []
                
                for n_features in feature_counts:
                    subset = method_data[method_data['n_features'] == n_features]
                    mse_means.append(subset['mse'].mean())
                    mse_stds.append(subset['std'].mean())
                
                method_names = ['Kendall排名', 'Spearman排名', '距离相关排名', 'MIC排名', '互信息排名',
                               'RF重要性', 'MDA得分', 'XGB重要性', 'LR系数']
                ax.errorbar(feature_counts, mse_means, yerr=mse_stds, 
                          label=method_names[j], color=colors[j], marker='o', capsize=5)
        
        ax.set_title(f'{dist} 距离')
        ax.set_xlabel('特征数量')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if data_type.lower() == 'mad':
        plt.savefig('results/mse_comparison_by_distance.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('results_vad/mse_comparison_by_distance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 固定使用Random Forest，比较9种特征选择方法
    if plt is None:
        return
        
    plt.figure(figsize=(15, 10))
    
    # 只使用Random Forest的数据
    rf_data = comparison_df[comparison_df['model_type'] == 'rf']
    
    # 定义9种特征选择方法及其颜色和中文名称
    methods = [
        'kendall_rank', 'spearman_rank', 'dcor_rank', 'mic_rank', 'mi_rank',
        'rf_importance', 'mda_score', 'xgb_importance', 'lr_coefficient'
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    method_names = [
        'Kendall排名', 'Spearman排名', '距离相关排名', 'MIC排名', '互信息排名',
        'RF重要性', 'MDA得分', 'XGB重要性', 'LR系数'
    ]
    
    for j, method in enumerate(methods):
        method_data = rf_data[rf_data['method'] == method]
        if not method_data.empty:
            # 按特征数量分组
            feature_counts = sorted(method_data['n_features'].unique())
            mse_means = []
            
            for n_features in feature_counts:
                subset = method_data[method_data['n_features'] == n_features]
                mse_means.append(subset['mse'].mean())
            
            plt.plot(feature_counts, mse_means, 
                    label=f'{method_names[j]}', 
                    color=colors[j], marker='o', linestyle='-', linewidth=2, markersize=6)
    
    plt.title(f'{data_type.upper()}数据Random Forest模型下9种特征选择方法的MSE比较', fontsize=16)
    plt.xlabel('特征数量', fontsize=14)
    plt.ylabel('MSE (Mean Squared Error)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if data_type.lower() == 'mad':
        plt.savefig('results/mse_comparison_by_distance.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('results_vad/mse_comparison_by_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_mse_report(comparison_df, data_type='mad'):
    """生成MSE分析报告"""
    print("正在生成MSE分析报告...")
    
    report = []
    report.append(f"=== {data_type.upper()}数据MSE比较分析报告 ===\n")
    
    # 1. 总体MSE统计
    report.append("1. 总体MSE统计:")
    report.append(f"   总测试组合数: {len(comparison_df)}")
    report.append(f"   平均MSE: {comparison_df['mse'].mean():.6f}")
    report.append(f"   MSE标准差: {comparison_df['mse'].std():.6f}")
    report.append(f"   最小MSE: {comparison_df['mse'].min():.6f}")
    report.append(f"   最大MSE: {comparison_df['mse'].max():.6f}\n")
    
    # 2. 按方法比较
    report.append("2. 不同特征选择方法的MSE比较:")
    method_stats = comparison_df.groupby('method')['mse'].agg(['mean', 'std', 'min', 'max'])
    
    # 方法名称映射
    method_name_map = {
        'kendall_rank': 'Kendall排名',
        'spearman_rank': 'Spearman排名', 
        'dcor_rank': '距离相关排名',
        'mic_rank': 'MIC排名',
        'mi_rank': '互信息排名',
        'rf_importance': 'RF重要性',
        'mda_score': 'MDA得分',
        'xgb_importance': 'XGB重要性',
        'lr_coefficient': 'LR系数'
    }
    
    for method, stats in method_stats.iterrows():
        method_name = method_name_map.get(method, method)
        report.append(f"   {method_name}:")
        report.append(f"     平均MSE: {stats['mean']:.6f}")
        report.append(f"     MSE标准差: {stats['std']:.6f}")
        report.append(f"     最小MSE: {stats['min']:.6f}")
        report.append(f"     最大MSE: {stats['max']:.6f}")
    report.append("")
    
    # 3. 按模型比较
    report.append("3. 不同模型的MSE比较:")
    model_stats = comparison_df.groupby('model_type')['mse'].agg(['mean', 'std', 'min', 'max'])
    for model, stats in model_stats.iterrows():
        report.append(f"   {model}:")
        report.append(f"     平均MSE: {stats['mean']:.6f}")
        report.append(f"     MSE标准差: {stats['std']:.6f}")
        report.append(f"     最小MSE: {stats['min']:.6f}")
        report.append(f"     最大MSE: {stats['max']:.6f}")
    report.append("")
    
    # 4. 最佳组合
    report.append("4. 最佳MSE组合:")
    best_idx = comparison_df['mse'].idxmin()
    best_result = comparison_df.loc[best_idx]
    method_name = method_name_map.get(best_result['method'], best_result['method'])
    report.append(f"   距离: {best_result['distance']}")
    report.append(f"   方法: {method_name}")
    report.append(f"   特征数量: {best_result['n_features']}")
    report.append(f"   模型: {best_result['model_type']}")
    report.append(f"   MSE: {best_result['mse']:.6f}")
    report.append(f"   标准差: {best_result['std']:.6f}\n")
    
    # 5. 按距离的最佳结果
    report.append("5. 各距离下的最佳结果:")
    for dist in ['5mm', '10mm', '15mm', '20mm']:
        dist_data = comparison_df[comparison_df['distance'] == dist]
        if not dist_data.empty:
            best_dist_idx = dist_data['mse'].idxmin()
            best_dist_result = dist_data.loc[best_dist_idx]
            method_name = method_name_map.get(best_dist_result['method'], best_dist_result['method'])
            report.append(f"   {dist}:")
            report.append(f"     方法: {method_name}")
            report.append(f"     特征数量: {best_dist_result['n_features']}")
            report.append(f"     模型: {best_dist_result['model_type']}")
            report.append(f"     MSE: {best_dist_result['mse']:.6f}")
    
    # 保存报告
    if data_type.lower() == 'mad':
        with open('results/mse_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    else:
        with open('results_vad/mse_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    
    # 打印报告
    for line in report:
        print(line)
    
    return report

def main(data_type='mad'):
    """主函数"""
    print(f"=== {data_type.upper()}数据MSE比较分析 ===")
    
    # 1. 加载特征排序结果
    feature_rankings = load_feature_rankings(data_type)
    
    # 2. 加载原始数据
    datasets = load_original_data(data_type)
    
    # 3. 比较不同特征选择方法
    results = compare_feature_selection_methods(datasets, feature_rankings)
    
    # 4. 创建MSE比较表
    comparison_df = create_mse_comparison_table(results, data_type)
    
    # 5. 可视化比较结果
    visualize_mse_comparison(comparison_df, data_type)
    
    # 6. 生成分析报告
    generate_mse_report(comparison_df, data_type)
    
    print(f"\n=== {data_type.upper()}数据MSE比较分析完成 ===")
    if data_type.lower() == 'mad':
        print("结果已保存到 results/ 目录")
    else:
        print("结果已保存到 results_vad 目录")

if __name__ == "__main__":
    main()