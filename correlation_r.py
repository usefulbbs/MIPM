from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy import stats
import pandas as pd
def correlation_analysis(datasets):
    """分析每个距离下特征与目标变量的相关性"""
    print("\n正在进行相关性分析...")
    
    correlation_results = {}
    
    for dist, data_dict in datasets.items():
        print(f"\n分析 {dist} 距离的数据...")
        
        X = data_dict['features']
        y = data_dict['target']
        
        # 计算Kendall's Tau相关系数
        kendall_corr = []
        for col in X.columns:
            corr, p_value = stats.kendalltau(X[col], y)
            kendall_corr.append({
                'feature': col,
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            })
        
        # 按绝对相关系数排序
        kendall_df = pd.DataFrame(kendall_corr)
        kendall_df = kendall_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        
        # 计算Spearman相关系数
        spearman_corr = []
        for col in X.columns:
            corr, p_value = stats.spearmanr(X[col], y)
            spearman_corr.append({
                'feature': col,
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            })
        
        spearman_df = pd.DataFrame(spearman_corr)
        spearman_df = spearman_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        
        # 计算距离相关性
        dcor_scores = distance_correlation(X, y)
        dcor_df = pd.DataFrame({
            'feature': X.columns,
            'distance_correlation': dcor_scores
        }).sort_values('distance_correlation', ascending=False).reset_index(drop=True)
        
        # 计算最大信息系数(MIC)
        mic_scores = max_information_coefficient(X, y)
        mic_df = pd.DataFrame({
            'feature': X.columns,
            'mic': mic_scores
        }).sort_values('mic', ascending=False).reset_index(drop=True)
        
        # 计算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False).reset_index(drop=True)
        
        correlation_results[dist] = {
            'kendall': kendall_df,
            'spearman': spearman_df,
            'distance_correlation': dcor_df,
            'mic': mic_df,
            'mutual_info': mi_df
        }
        
        print(f"  Kendall's Tau相关系数最高的特征: {kendall_df.iloc[0]['feature']} (τ={kendall_df.iloc[0]['correlation']:.4f})")
        print(f"  Spearman相关系数最高的特征: {spearman_df.iloc[0]['feature']} (ρ={spearman_df.iloc[0]['correlation']:.4f})")
        print(f"  距离相关性最高的特征: {dcor_df.iloc[0]['feature']} (dCor={dcor_df.iloc[0]['distance_correlation']:.4f})")
        print(f"  MIC最高的特征: {mic_df.iloc[0]['feature']} (MIC={mic_df.iloc[0]['mic']:.4f})")
        print(f"  互信息最高的特征: {mi_df.iloc[0]['feature']} (MI={mi_df.iloc[0]['mutual_info']:.4f})")
        
        # 保存前20个最相关特征到CSV文件
        save_top_features_to_csv(dist, kendall_df, spearman_df, dcor_df, mic_df, mi_df)
    
    return correlation_results

def save_top_features_to_csv(dist, kendall_df, spearman_df, dcor_df, mic_df, mi_df):
    """保存每个距离下所有特征的综合排名到CSV文件"""
    # 创建综合排名表（包含所有特征）
    all_features_ranking = []
    
    # 获取所有特征名称
    all_features = dcor_df['feature'].tolist()
    
    for feature in all_features:
        # 获取该特征在各个指标下的排名和值
        kendall_rank = kendall_df[kendall_df['feature'] == feature].index[0] + 1 if feature in kendall_df['feature'].values else len(kendall_df)
        spearman_rank = spearman_df[spearman_df['feature'] == feature].index[0] + 1 if feature in spearman_df['feature'].values else len(spearman_df)
        dcor_rank = dcor_df[dcor_df['feature'] == feature].index[0] + 1 if feature in dcor_df['feature'].values else len(dcor_df)
        mic_rank = mic_df[mic_df['feature'] == feature].index[0] + 1 if feature in mic_df['feature'].values else len(mic_df)
        mi_rank = mi_df[mi_df['feature'] == feature].index[0] + 1 if feature in mi_df['feature'].values else len(mi_df)
        
        # 获取相关系数值
        kendall_corr = kendall_df[kendall_df['feature'] == feature]['correlation'].iloc[0] if feature in kendall_df['feature'].values else 0
        spearman_corr = spearman_df[spearman_df['feature'] == feature]['correlation'].iloc[0] if feature in spearman_df['feature'].values else 0
        dcor_corr = dcor_df[dcor_df['feature'] == feature]['distance_correlation'].iloc[0] if feature in dcor_df['feature'].values else 0
        mic_corr = mic_df[mic_df['feature'] == feature]['mic'].iloc[0] if feature in mic_df['feature'].values else 0
        mi_corr = mi_df[mi_df['feature'] == feature]['mutual_info'].iloc[0] if feature in mi_df['feature'].values else 0
        
        # 计算综合得分（排名越靠前得分越高，前50名有得分）
        kendall_score = 50 - kendall_rank + 1 if kendall_rank <= 50 else 0
        spearman_score = 50 - spearman_rank + 1 if spearman_rank <= 50 else 0
        dcor_score = 50 - dcor_rank + 1 if dcor_rank <= 50 else 0
        mic_score = 50 - mic_rank + 1 if mic_rank <= 50 else 0
        mi_score = 50 - mi_rank + 1 if mi_rank <= 50 else 0
        
        total_score = kendall_score + spearman_score + dcor_score + mic_score + mi_score
        
        all_features_ranking.append({
            'feature': feature,
            'kendall_rank': kendall_rank,
            'kendall_correlation': kendall_corr,
            'kendall_score': kendall_score,
            'spearman_rank': spearman_rank,
            'spearman_correlation': spearman_corr,
            'spearman_score': spearman_score,
            'dcor_rank': dcor_rank,
            'dcor_correlation': dcor_corr,
            'dcor_score': dcor_score,
            'mic_rank': mic_rank,
            'mic_value': mic_corr,
            'mic_score': mic_score,
            'mi_rank': mi_rank,
            'mi_value': mi_corr,
            'mi_score': mi_score,
            'total_score': total_score
        })
    
    # 按综合得分排序
    ranking_df = pd.DataFrame(all_features_ranking)
    ranking_df = ranking_df.sort_values('total_score', ascending=False)
    
    # 保存完整排名表到CSV文件
    filename_full = f'feature_ranking_{dist}.csv'
    ranking_df.to_csv(filename_full, index=False, encoding='utf-8-sig')
    print(f"  完整特征排名表已保存到: {filename_full}")
    
    # 保存前20个特征到单独的CSV文件
    top_20_df = ranking_df.head(20)
    filename_top20 = f'top_20_features_{dist}.csv'
    top_20_df.to_csv(filename_top20, index=False, encoding='utf-8-sig')
    print(f"  前20个最相关特征已保存到: {filename_top20}")
    
    # 打印前10个特征的综合排名
    print(f"  {dist} 距离下综合排名前10的特征:")
    for i, row in ranking_df.head(10).iterrows():
        print(f"    {row['feature']}: 综合得分={row['total_score']}, Kendall排名={row['kendall_rank']}, Spearman排名={row['spearman_rank']}, dCor排名={row['dcor_rank']}, MIC排名={row['mic_rank']}, MI排名={row['mi_rank']}")
    
    return ranking_df
def distance_correlation(X, y):
    """计算距离相关性"""
    def dcor(x, y):
        n = len(x)
        if n < 4:
            return 0.0
        
        # 计算距离矩阵
        x_dist = squareform(pdist(x.reshape(-1, 1)))
        y_dist = squareform(pdist(y.reshape(-1, 1)))
        
        # 计算双中心距离
        x_mean = x_dist.mean(axis=0)
        y_mean = y_dist.mean(axis=0)
        x_grand_mean = x_dist.mean()
        y_grand_mean = y_dist.mean()
        
        x_centered = x_dist - x_mean.reshape(-1, 1) - x_mean.reshape(1, -1) + x_grand_mean
        y_centered = y_dist - y_mean.reshape(-1, 1) - y_mean.reshape(1, -1) + y_grand_mean
        
        # 计算距离相关性
        numerator = (x_centered * y_centered).sum()
        denominator_x = (x_centered ** 2).sum()
        denominator_y = (y_centered ** 2).sum()
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        return numerator / np.sqrt(denominator_x * denominator_y)
    
    # 对每个特征计算距离相关性
    dcor_scores = []
    for col in X.columns:
        try:
            dcor_score = dcor(X[col].values, y.values)
            dcor_scores.append(dcor_score)
        except:
            dcor_scores.append(0.0)
    
    return dcor_scores

def max_information_coefficient(X, y, max_ticks=20):
    """计算最大信息系数(MIC)"""
    def mic(x, y):
        n = len(x)
        if n < 4:
            return 0.0
        
        # 简化的MIC计算（完整实现需要minepy库）
        # 这里使用网格化的互信息近似
        x_bins = min(max_ticks, int(np.sqrt(n)))
        y_bins = min(max_ticks, int(np.sqrt(n)))
        
        try:
            # 创建2D直方图
            hist_2d, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
            hist_x, _ = np.histogram(x, bins=x_bins)
            hist_y, _ = np.histogram(y, bins=y_bins)
            
            # 计算互信息
            n_total = hist_2d.sum()
            if n_total == 0:
                return 0.0
            
            mi = 0.0
            for i in range(x_bins):
                for j in range(y_bins):
                    if hist_2d[i, j] > 0:
                        p_xy = hist_2d[i, j] / n_total
                        p_x = hist_x[i] / n_total
                        p_y = hist_y[j] / n_total
                        if p_x > 0 and p_y > 0:
                            mi += p_xy * np.log2(p_xy / (p_x * p_y))
            
            # 归一化到[0,1]
            max_mi = min(np.log2(x_bins), np.log2(y_bins))
            if max_mi > 0:
                return mi / max_mi
            else:
                return 0.0
                
        except:
            return 0.0
    
    # 对每个特征计算MIC
    mic_scores = []
    for col in X.columns:
        try:
            mic_score = mic(X[col].values, y.values)
            mic_scores.append(mic_score)
        except:
            mic_scores.append(0.0)
    
    return mic_scores
