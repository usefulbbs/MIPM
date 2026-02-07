import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_feature_rankings(data_type='mad'):
    """加载特征排序结果"""
    print(f"正在加载{data_type.upper()}数据特征排序结果...")
    
    rankings = {}
    
    # 根据数据类型选择路径
    if data_type.lower() == 'mad':
        base_path = 'results_mad'
    elif data_type.lower() == 'vad':
        base_path = 'results_vad'
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    # 加载相关性分析的特征排序
    print("  加载相关性分析特征排序...")
    correlation_rankings = {}
    distances = ['5mm', '10mm', '15mm', '20mm']
    
    for dist in distances:
        try:
            file_path = f'{base_path}/correlation/feature_ranking_{dist}.csv'
            df = pd.read_csv(file_path)
            correlation_rankings[dist] = df
            print(f"    已加载 {dist} 距离相关性排序: {len(df)} 个特征")
        except FileNotFoundError:
            print(f"    警告: 未找到 {dist} 距离的相关性排序文件: {file_path}")
    
    # 加载模型特征重要性
    print("  加载模型特征重要性...")
    model_rankings = {}
    
    for dist in distances:
        try:
            file_path = f'{base_path}/model/feature_importance_{dist}.csv'
            df = pd.read_csv(file_path)
            model_rankings[dist] = df
            print(f"    已加载 {dist} 距离模型重要性: {len(df)} 个特征")
        except FileNotFoundError:
            print(f"    警告: 未找到 {dist} 距离的模型重要性文件: {file_path}")
    
    rankings = {
        'correlation': correlation_rankings,
        'model': model_rankings
    }
    
    return rankings

def load_original_data(data_type='mad'):
    """加载原始数据"""
    print(f"正在加载{data_type.upper()}数据...")
    
    datasets = {}
    distances = ['5mm', '10mm', '15mm', '20mm']
    
    for dist in distances:
        if data_type.lower() == 'mad':
            file_path = f'dataset_mad/mad_{dist}.xlsx'
            target_col = 'Mad'
        elif data_type.lower() == 'vad':
            file_path = f'dataset_vad/vad_{dist}.xlsx'
            target_col = 'Vad'
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        df = pd.read_excel(file_path)
        
        # 重命名列
        df.columns = ['sample_name', target_col] + [f'feature_{i}' for i in range(len(df.columns)-2)]
        
        # 分离特征和目标变量
        X = df.iloc[:, 2:]  # 特征列
        y = df[target_col]  # 目标变量
        
        datasets[dist] = {
            'data': df,
            'features': X,
            'target': y,
            'feature_names': X.columns.tolist()
        }
        
        print(f"  {dist}: {X.shape[1]} 个特征, {len(y)} 个样本")
    
    return datasets

def verify_feature_selection(datasets, feature_rankings):
    """验证三种方法选择的特征"""
    print("\n=== 验证三种方法选择的特征 ===")
    
    for dist in datasets.keys():
        print(f"\n{dist} 距离的特征选择验证:")
        
        X = datasets[dist]['features']
        y = datasets[dist]['target']
        
        # 获取特征名称
        feature_names = X.columns.tolist()
        
        # 方法1: 相关性分析综合方法
        print("\n1. 相关性分析综合方法:")
        if 'correlation' in feature_rankings and dist in feature_rankings['correlation']:
            ranking_df = feature_rankings['correlation'][dist]
            
            # 检查是否有total_score列
            if 'total_score' in ranking_df.columns:
                print("  使用 total_score 列")
                # 按total_score排序，选择前20个
                top_correlation = ranking_df.nlargest(20, 'total_score')
                print("  前20个特征:")
                for i, row in top_correlation.iterrows():
                    print(f"    {row['feature']}: total_score={row['total_score']:.4f}")
            else:
                print("  未找到 total_score 列，可用列名:")
                print(f"    {ranking_df.columns.tolist()}")
        
        # 方法2: 模型综合方法
        print("\n2. 模型综合方法:")
        if 'model' in feature_rankings and dist in feature_rankings['model']:
            ranking_df = feature_rankings['model'][dist]
            
            # 检查是否有combined_score列
            if 'combined_score' in ranking_df.columns:
                print("  使用 combined_score 列")
                # 按combined_score排序，选择前20个
                top_model = ranking_df.nlargest(20, 'combined_score')
                print("  前20个特征:")
                for i, row in top_model.iterrows():
                    print(f"    {row['feature']}: combined_score={row['combined_score']:.4f}")
            else:
                print("  未找到 combined_score 列，可用列名:")
                print(f"    {ranking_df.columns.tolist()}")
        
        # 方法3: 两者结合的综合方法
        print("\n3. 两者结合的综合方法:")
        if ('correlation' in feature_rankings and dist in feature_rankings['correlation'] and
            'model' in feature_rankings and dist in feature_rankings['model']):
            
            corr_df = feature_rankings['correlation'][dist]
            model_df = feature_rankings['model'][dist]
            
            # 创建特征索引映射
            feature_to_index = {name: i for i, name in enumerate(feature_names)}
            
            # 计算综合得分
            combined_scores = np.zeros(len(feature_names))
            
            if 'total_score' in corr_df.columns and 'combined_score' in model_df.columns:
                print("  计算综合得分 (correlation + model，排除MI和LR方法)")
                
                # 为每个特征计算综合得分（排除MI和LR方法）
                for _, row in corr_df.iterrows():
                    feature_name = row['feature']
                    if feature_name in feature_to_index:
                        idx = feature_to_index[feature_name]
                        # 计算不包含MI的得分：kendall_score + spearman_score + dcor_score + mic_score
                        if 'kendall_score' in row and 'spearman_score' in row and 'dcor_score' in row and 'mic_score' in row:
                            score_without_mi = row['kendall_score'] + row['spearman_score'] + row['dcor_score'] + row['mic_score']
                            combined_scores[idx] += score_without_mi
                        else:
                            # 如果没有单独的得分列，使用total_score但需要重新计算
                            combined_scores[idx] += row['total_score']
                
                for _, row in model_df.iterrows():
                    feature_name = row['feature']
                    if feature_name in feature_to_index:
                        idx = feature_to_index[feature_name]
                        # 计算不包含LR的得分：rf_importance_norm + mda_score_norm + xgb_importance_norm
                        if 'rf_importance_norm' in row and 'mda_score_norm' in row and 'xgb_importance_norm' in row:
                            score_without_lr = row['rf_importance_norm'] + row['mda_score_norm'] + row['xgb_importance_norm']
                            combined_scores[idx] += score_without_lr
                        else:
                            # 如果没有单独的得分列，使用combined_score但需要重新计算
                            combined_scores[idx] += row['combined_score']
                
                # 选择综合得分最高的前20个特征
                top_indices = np.argsort(combined_scores)[-20:][::-1]
                print("  前20个特征:")
                for i, idx in enumerate(top_indices):
                    feature_name = feature_names[idx]
                    score = combined_scores[idx]
                    print(f"    {feature_name}: combined_score={score:.4f}")
            else:
                print("  缺少必要的得分列，无法计算综合得分")
        
        print("\n" + "="*50)

def calculate_mse_for_feature_counts(datasets, feature_rankings, data_type='mad'):
    """计算不同特征数量下的MSE损失"""
    print("\n=== 计算不同特征数量下的MSE损失 ===")
    
    # 定义要测试的特征数量
    feature_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # 存储结果
    results = {
        'distances': [],
        'feature_counts': [],
        'correlation_mse': [],
        'model_mse': [],
        'combined_mse': []
    }
    
    for dist in datasets.keys():
        print(f"\n正在处理 {dist} 距离...")
        
        X = datasets[dist]['features']
        y = datasets[dist]['target']
        
        # 获取特征名称
        feature_names = X.columns.tolist()
        
        # 检查是否有必要的排序文件
        if ('correlation' not in feature_rankings or dist not in feature_rankings['correlation'] or
            'model' not in feature_rankings or dist not in feature_rankings['model']):
            print(f"  跳过 {dist} 距离，缺少必要的排序文件")
            continue
        
        corr_df = feature_rankings['correlation'][dist]
        model_df = feature_rankings['model'][dist]
        
        # 检查必要的列
        if 'total_score' not in corr_df.columns or 'combined_score' not in model_df.columns:
            print(f"  跳过 {dist} 距离，缺少必要的得分列")
            continue
        
        # 为每个特征数量计算MSE
        for n_features in feature_counts:
            print(f"    测试 {n_features} 个特征...")
            
            # 方法1: 相关性分析
            top_corr_features = corr_df.nlargest(n_features, 'total_score')['feature'].tolist()
            X_corr = X[top_corr_features]
            
            # 方法2: 模型重要性
            top_model_features = model_df.nlargest(n_features, 'combined_score')['feature'].tolist()
            X_model = X[top_model_features]
            
            # 方法3: 两者结合（排除MI方法和LR方法）
            # 创建特征索引映射
            feature_to_index = {name: i for i, name in enumerate(feature_names)}
            
            # 计算综合得分（排除MI和LR方法）
            combined_scores = np.zeros(len(feature_names))
            
            # 相关性分析得分（排除MI方法）
            for _, row in corr_df.iterrows():
                feature_name = row['feature']
                if feature_name in feature_to_index:
                    idx = feature_to_index[feature_name]
                    # 计算不包含MI的得分：kendall_score + spearman_score + dcor_score + mic_score
                    if 'kendall_score' in row and 'spearman_score' in row and 'dcor_score' in row and 'mic_score' in row:
                        score_without_mi = row['kendall_score'] + row['spearman_score'] + row['dcor_score'] + row['mic_score']
                        combined_scores[idx] += score_without_mi
                    else:
                        # 如果没有单独的得分列，使用total_score但需要重新计算
                        combined_scores[idx] += row['total_score']
            
            # 模型重要性得分（排除LR方法）
            for _, row in model_df.iterrows():
                feature_name = row['feature']
                if feature_name in feature_to_index:
                    idx = feature_to_index[feature_name]
                    # 计算不包含LR的得分：rf_importance_norm + mda_score_norm + xgb_importance_norm
                    if 'rf_importance_norm' in row and 'mda_score_norm' in row and 'xgb_importance_norm' in row:
                        score_without_lr = row['rf_importance_norm'] + row['mda_score_norm'] + row['xgb_importance_norm']
                        combined_scores[idx] += score_without_lr
                    else:
                        # 如果没有单独的归一化列，使用combined_score但需要重新计算
                        combined_scores[idx] += row['combined_score']
            
            # 选择综合得分最高的前n_features个特征
            top_indices = np.argsort(combined_scores)[-n_features:][::-1]
            top_combined_features = [feature_names[i] for i in top_indices]
            X_combined = X[top_combined_features]
            
            # 计算MSE（使用交叉验证）
            try:
                # 相关性方法MSE
                rf_corr = RandomForestRegressor(n_estimators=100, random_state=42)
                mse_corr = -cross_val_score(rf_corr, X_corr, y, cv=5, scoring='neg_mean_squared_error').mean()
                
                # 模型方法MSE
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                mse_model = -cross_val_score(rf_model, X_model, y, cv=5, scoring='neg_mean_squared_error').mean()
                
                # 综合方法MSE
                rf_combined = RandomForestRegressor(n_estimators=100, random_state=42)
                mse_combined = -cross_val_score(rf_combined, X_combined, y, cv=5, scoring='neg_mean_squared_error').mean()
                
                # 存储结果
                results['distances'].append(dist)
                results['feature_counts'].append(n_features)
                results['correlation_mse'].append(mse_corr)
                results['model_mse'].append(mse_model)
                results['combined_mse'].append(mse_combined)
                
                print(f"      MSE - 相关性: {mse_corr:.6f}, 模型: {mse_model:.6f}, 综合: {mse_combined:.6f}")
                
            except Exception as e:
                print(f"      计算MSE时出错: {e}")
                continue
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 根据数据类型选择保存路径
    if data_type.lower() == 'vad':
        save_path = 'results_vad/mse_comparison_by_distance.csv'
        # 确保目录存在
        os.makedirs('results_vad', exist_ok=True)
    else:
        save_path = 'results_mad/mse_comparison_by_distance.csv'
        # 确保目录存在
        os.makedirs('results_mad', exist_ok=True)
    
    # 保存结果
    results_df.to_csv(save_path, index=False)
    print(f"\n结果已保存到: {save_path}")
    
    return results_df

def visualize_mse_comparison(results_df, data_type='mad'):
    """可视化MSE比较结果"""
    print("\n=== 生成MSE比较可视化图表 ===")
    
    if results_df.empty:
        print("没有数据可以可视化")
        return
    
    # 根据数据类型选择标题和保存路径
    if data_type.lower() == 'mad':
        title = 'MAD数据不同距离下三种特征选择方法的MSE比较'
        save_path = 'results_mad/mse_comparison_by_distance.png'
        csv_path = 'results_mad/mse_summary_by_distance.csv'
        # 确保目录存在
        os.makedirs('results_mad', exist_ok=True)
    elif data_type.lower() == 'vad':
        title = 'VAD数据不同距离下三种特征选择方法的MSE比较'
        save_path = 'results_vad/mse_comparison_by_distance.png'
        csv_path = 'results_vad/mse_summary_by_distance.csv'
        # 确保目录存在
        os.makedirs('results_vad', exist_ok=True)
    else:
        title = '不同距离下三种特征选择方法的MSE比较'
        save_path = 'results_mad/mse_comparison_by_distance.png'
        csv_path = 'results_mad/mse_summary_by_distance.csv'
        # 确保目录存在
        os.makedirs('results_mad', exist_ok=True)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    distances = results_df['distances'].unique()
    
    for i, dist in enumerate(distances):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # 获取当前距离的数据
        dist_data = results_df[results_df['distances'] == dist]
        
        if dist_data.empty:
            continue
        
        # 绘制三条线
        ax.plot(dist_data['feature_counts'], dist_data['correlation_mse'], 
                'o-', label='相关性分析', linewidth=2, markersize=6)
        ax.plot(dist_data['feature_counts'], dist_data['model_mse'], 
                's-', label='模型重要性', linewidth=2, markersize=6)
        ax.plot(dist_data['feature_counts'], dist_data['combined_mse'], 
                '^-', label='两者结合', linewidth=2, markersize=6)
        
        ax.set_title(f'{dist} 距离')
        ax.set_xlabel('特征数量')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围，使图表更清晰
        all_mse = np.concatenate([
            dist_data['correlation_mse'].values,
            dist_data['model_mse'].values,
            dist_data['combined_mse'].values
        ])
        if len(all_mse) > 0:
            y_min, y_max = all_mse.min(), all_mse.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建汇总表格
    print("\n=== MSE比较汇总表 ===")
    summary_data = []
    
    for dist in distances:
        dist_data = results_df[results_df['distances'] == dist]
        if not dist_data.empty:
            for n_features in dist_data['feature_counts'].unique():
                feature_data = dist_data[dist_data['feature_counts'] == n_features]
                if not feature_data.empty:
                    summary_data.append({
                        '距离': dist,
                        '特征数量': n_features,
                        '相关性MSE': feature_data['correlation_mse'].iloc[0],
                        '模型MSE': feature_data['model_mse'].iloc[0],
                        '综合MSE': feature_data['combined_mse'].iloc[0]
                    })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 保存汇总表
    summary_df.to_csv(csv_path, index=False)
    print(f"\n汇总表已保存到: {csv_path}")

def main(data_type='mad'):
    """主函数"""
    print(f"=== {data_type.upper()}数据不同距离特征选择MSE损失比较程序 ===")
    
    # 1. 加载特征排序结果
    feature_rankings = load_feature_rankings(data_type)
    
    # 2. 加载原始数据
    datasets = load_original_data(data_type)
    
    # 3. 验证特征选择
    verify_feature_selection(datasets, feature_rankings)
    
    # 4. 计算不同特征数量下的MSE
    results_df = calculate_mse_for_feature_counts(datasets, feature_rankings, data_type)
    
    # 5. 可视化MSE比较结果
    if not results_df.empty:
        visualize_mse_comparison(results_df, data_type)
    else:
        print("\n没有可用的结果数据")

if __name__ == "__main__":
    main()