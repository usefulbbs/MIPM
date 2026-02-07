import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def calculate_mda(model, X, y, n_repeats=10):
    """
    计算MDA (Mean Decrease in Accuracy)
    通过打乱特征值来评估特征重要性
    """
    from sklearn.metrics import r2_score
    
    # 获取基准性能
    baseline_score = r2_score(y, model.predict(X))
    mda_scores = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        feature_importance = 0
        
        for _ in range(n_repeats):
            # 保存原始特征值
            original_values = X.iloc[:, i].copy()
            
            # 打乱特征值
            X_shuffled = X.copy()
            X_shuffled.iloc[:, i] = np.random.permutation(original_values)
            
            # 计算打乱后的性能
            shuffled_score = r2_score(y, model.predict(X_shuffled))
            
            # 性能下降程度
            feature_importance += (baseline_score - shuffled_score)
        
        # 平均重要性
        mda_scores[i] = feature_importance / n_repeats
    
    return mda_scores

def feature_importance_analysis(datasets):
    """使用机器学习模型计算特征重要性"""
    print("\n正在进行基于模型的特征重要性分析...")
    
    # 检查必要的库
    try:
        import xgboost as xgb
    except ImportError:
        print("缺少XGBoost库，请安装: pip install xgboost")
        return {}
    
    importance_results = {}
    
    for dist, data_dict in datasets.items():
        print(f"\n分析 {dist} 距离的数据...")
        
        X = data_dict['features']
        y = data_dict['target']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 1. 随机森林特征重要性
        print("  训练随机森林模型...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # 使用交叉验证评估模型性能
        rf_cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
        print(f"  随机森林交叉验证R²分数: {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
        
        # 训练最终模型并获取特征重要性
        rf_model.fit(X_scaled, y)
        rf_importance = rf_model.feature_importances_
        
        # 计算MDA (Mean Decrease in Accuracy)
        print("  计算MDA (Mean Decrease in Accuracy)...")
        mda_scores = calculate_mda(rf_model, X_scaled_df, y, n_repeats=10)
        
        # 2. XGBoost特征重要性
        print("  训练XGBoost模型...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # 使用交叉验证评估模型性能
        xgb_cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=5, scoring='r2')
        print(f"  XGBoost交叉验证R²分数: {xgb_cv_scores.mean():.4f} (±{xgb_cv_scores.std():.4f})")
        
        # 训练最终模型并获取特征重要性
        xgb_model.fit(X_scaled, y)
        xgb_importance = xgb_model.feature_importances_
        
        # 3. 线性回归系数（作为特征重要性的参考）
        print("  训练线性回归模型...")
        lr_model = LinearRegression()
        lr_cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='r2')
        print(f"  线性回归交叉验证R²分数: {lr_cv_scores.mean():.4f} (±{lr_cv_scores.std():.4f})")
        
        # 训练最终模型并获取系数
        lr_model.fit(X_scaled, y)
        lr_coefficients = np.abs(lr_model.coef_)
        
        # 4. 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf_importance,
            'mda_score': mda_scores,
            'xgb_importance': xgb_importance,
            'lr_coefficient': lr_coefficients
        })
        
        # 计算综合重要性得分
        # 标准化各个重要性指标到[0,1]范围
        importance_df['rf_importance_norm'] = (importance_df['rf_importance'] - importance_df['rf_importance'].min()) / (importance_df['rf_importance'].max() - importance_df['rf_importance'].min())
        importance_df['mda_score_norm'] = (importance_df['mda_score'] - importance_df['mda_score'].min()) / (importance_df['mda_score'].max() - importance_df['mda_score'].min())
        importance_df['xgb_importance_norm'] = (importance_df['xgb_importance'] - importance_df['xgb_importance'].min()) / (importance_df['xgb_importance'].max() - importance_df['xgb_importance'].min())
        importance_df['lr_coefficient_norm'] = (importance_df['lr_coefficient'] - importance_df['lr_coefficient'].min()) / (importance_df['lr_coefficient'].max() - importance_df['lr_coefficient'].min())
        
        # 计算综合得分（加权平均）
        importance_df['combined_score'] = (
            importance_df['rf_importance_norm'] * 0.3 +
            importance_df['mda_score_norm'] * 0.3 +
            importance_df['xgb_importance_norm'] * 0.3 +
            importance_df['lr_coefficient_norm'] * 0.1
        )
        
        # 按综合得分排序
        importance_df = importance_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        
        # 保存特征重要性结果
        importance_results[dist] = {
            'importance_df': importance_df,
            'rf_model': rf_model,
            'xgb_model': xgb_model,
            'lr_model': lr_model,
            'rf_cv_score': rf_cv_scores.mean(),
            'xgb_cv_score': xgb_cv_scores.mean(),
            'lr_cv_score': lr_cv_scores.mean()
        }
        
        # 打印前10个最重要的特征
        print(f"  {dist} 距离下前10个最重要的特征:")
        for i, row in importance_df.head(10).iterrows():
            print(f"    {i+1:2d}. {row['feature']}: 综合得分={row['combined_score']:.4f}, RF={row['rf_importance']:.4f}, MDA={row['mda_score']:.4f}, XGB={row['xgb_importance']:.4f}")
        
        # 保存特征重要性到CSV文件
        save_feature_importance_to_csv(dist, importance_df)
    
    return importance_results

def save_feature_importance_to_csv(dist, importance_df):
    """保存特征重要性结果到CSV文件"""
    # 保存完整特征重要性表
    filename_full = f'feature_importance_{dist}.csv'
    importance_df.to_csv(filename_full, index=False, encoding='utf-8-sig')
    print(f"  特征重要性表已保存到: {filename_full}")
    
    # 保存前20个最重要特征
    top_20_df = importance_df.head(20)
    filename_top20 = f'top_20_important_features_{dist}.csv'
    top_20_df.to_csv(filename_top20, index=False, encoding='utf-8-sig')
    print(f"  前20个最重要特征已保存到: {filename_top20}")

def compare_correlation_and_importance(correlation_results, importance_results):
    """比较统计相关性和模型特征重要性的结果"""
    print("\n=== 统计相关性与模型特征重要性的对比分析 ===")
    
    for dist in correlation_results.keys():
        print(f"\n{dist} 距离下的对比分析:")
        
        # 获取统计相关性前10名
        stat_top10 = correlation_results[dist]['distance_correlation'].head(10)['feature'].tolist()
        
        # 获取模型重要性前10名
        model_top10 = importance_results[dist]['importance_df'].head(10)['feature'].tolist()
        
        print(f"  统计相关性前10名特征:")
        for i, feature in enumerate(stat_top10, 1):
            print(f"    {i:2d}. {feature}")
        
        print(f"  模型重要性前10名特征:")
        for i, feature in enumerate(model_top10, 1):
            print(f"    {i:2d}. {feature}")
        
        # 计算重叠度
        overlap = set(stat_top10) & set(model_top10)
        overlap_ratio = len(overlap) / 10
        print(f"  前10名特征重叠度: {len(overlap)}/10 ({overlap_ratio:.1%})")
        
        if overlap:
            print(f"  重叠的特征: {', '.join(sorted(overlap))}")
        
        # 分析排名差异
        print(f"  排名差异分析:")
        for feature in stat_top10[:5]:  # 只看统计相关性前5名
            if feature in model_top10:
                stat_rank = stat_top10.index(feature) + 1
                model_rank = model_top10.index(feature) + 1
                rank_diff = abs(stat_rank - model_rank)
                print(f"    {feature}: 统计排名={stat_rank}, 模型排名={model_rank}, 差异={rank_diff}")
            else:
                stat_rank = stat_top10.index(feature) + 1
                print(f"    {feature}: 统计排名={stat_rank}, 模型排名>10")

def feature_selection_final_recommendation(correlation_results, importance_results):
    """基于统计相关性和模型重要性的最终特征选择建议"""
    print("\n=== 最终特征选择建议 ===")
    
    for dist in correlation_results.keys():
        print(f"\n{dist} 距离下的最终建议:")
        
        # 获取统计相关性前20名
        stat_top20 = correlation_results[dist]['distance_correlation'].head(20)['feature'].tolist()
        
        # 获取模型重要性前20名
        model_top20 = importance_results[dist]['importance_df'].head(20)['feature'].tolist()
        
        # 计算综合排名（同时考虑统计相关性和模型重要性）
        feature_scores = {}
        for feature in set(stat_top20 + model_top20):
            score = 0
            # 统计相关性得分
            if feature in stat_top20:
                stat_rank = stat_top20.index(feature)
                score += (20 - stat_rank) * 0.5  # 权重0.5
            
            # 模型重要性得分
            if feature in model_top20:
                model_rank = model_top20.index(feature)
                score += (20 - model_rank) * 0.5  # 权重0.5
            
            feature_scores[feature] = score
        
        # 按综合得分排序
        final_ranking = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"  综合排名前15个特征:")
        for i, (feature, score) in enumerate(final_ranking[:15], 1):
            stat_rank = stat_top20.index(feature) + 1 if feature in stat_top20 else ">20"
            model_rank = model_top20.index(feature) + 1 if feature in model_top20 else ">20"
            print(f"    {i:2d}. {feature}: 综合得分={score:.1f}, 统计排名={stat_rank}, 模型排名={model_rank}")
        
        # 保存最终推荐特征
        final_features = [feature for feature, _ in final_ranking[:15]]
        final_df = pd.DataFrame({
            'rank': range(1, 16),
            'feature': final_features,
            'stat_rank': [stat_top20.index(f) + 1 if f in stat_top20 else 21 for f in final_features],
            'model_rank': [model_top20.index(f) + 1 if f in model_top20 else 21 for f in final_features]
        })
        
        filename_final = f'final_recommended_features_{dist}.csv'
        final_df.to_csv(filename_final, index=False, encoding='utf-8-sig')
        print(f"  最终推荐特征已保存到: {filename_final}")

def advanced_feature_selection(datasets, correlation_results, importance_results, n_features=20):
    """高级特征选择：结合多种方法选择最优特征子集"""
    print(f"\n=== 高级特征选择（选择{n_features}个特征）===")
    
    selection_results = {}
    
    for dist in datasets.keys():
        print(f"\n{dist} 距离下的高级特征选择:")
        
        # 获取各种方法的结果
        stat_df = correlation_results[dist]['distance_correlation']
        model_df = importance_results[dist]['importance_df']
        
        # 方法1：统计相关性排名
        stat_rank = stat_df.set_index('feature')['distance_correlation'].rank(ascending=False)
        
        # 方法2：模型重要性排名
        model_rank = model_df.set_index('feature')['combined_score'].rank(ascending=False)
        
        # 方法3：综合得分（统计相关性 + 模型重要性）
        combined_df = pd.DataFrame({
            'feature': stat_df['feature'],
            'stat_corr': stat_df['distance_correlation'],
            'model_importance': model_df.set_index('feature').loc[stat_df['feature']]['combined_score'].values,
            'stat_rank': stat_rank.values,
            'model_rank': model_rank.values
        })
        
        # 标准化得分
        combined_df['stat_corr_norm'] = (combined_df['stat_corr'] - combined_df['stat_corr'].min()) / (combined_df['stat_corr'].max() - combined_df['stat_corr'].min())
        combined_df['model_importance_norm'] = (combined_df['model_importance'] - combined_df['model_importance'].min()) / (combined_df['model_importance'].max() - combined_df['model_importance'].min())
        
        # 计算综合得分
        combined_df['final_score'] = (
            combined_df['stat_corr_norm'] * 0.4 +
            combined_df['model_importance_norm'] * 0.4 +
            (1 / combined_df['stat_rank']) * 0.1 +
            (1 / combined_df['model_rank']) * 0.1
        )
        
        # 按最终得分排序
        combined_df = combined_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        # 选择前n_features个特征
        selected_features = combined_df.head(n_features)
        
        print(f"  选择的{n_features}个特征:")
        for i, row in selected_features.iterrows():
            print(f"    {i+1:2d}. {row['feature']}: 最终得分={row['final_score']:.4f}, 统计相关性={row['stat_corr']:.4f}, 模型重要性={row['model_importance']:.4f}")
        
        # 保存选择结果
        filename_selected = f'selected_features_{n_features}_{dist}.csv'
        selected_features.to_csv(filename_selected, index=False, encoding='utf-8-sig')
        print(f"  选择的特征已保存到: {filename_selected}")
        
        selection_results[dist] = {
            'selected_features': selected_features,
            'combined_df': combined_df
        }
    
    return selection_results

def model_performance_with_selected_features(datasets, selection_results):
    """使用选择的特征评估模型性能"""
    print("\n=== 使用选择特征评估模型性能 ===")
    
    performance_results = {}
    
    for dist in datasets.keys():
        print(f"\n{dist} 距离下的模型性能评估:")
        
        # 获取原始数据和选择的特征
        X_full = datasets[dist]['features']
        y = datasets[dist]['target']
        selected_features = selection_results[dist]['selected_features']['feature'].tolist()
        
        # 使用选择的特征
        X_selected = X_full[selected_features]
        
        # 标准化特征
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X_full)
        X_selected_scaled = scaler.transform(X_selected)
        
        # 训练模型并比较性能
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        performance_comparison = {}
        
        for model_name, model in models.items():
            print(f"  评估 {model_name} 模型...")
            
            # 使用全部特征
            full_scores = cross_val_score(model, X_full_scaled, y, cv=5, scoring='r2')
            full_r2 = full_scores.mean()
            full_std = full_scores.std()
            
            # 使用选择的特征
            selected_scores = cross_val_score(model, X_selected_scaled, y, cv=5, scoring='r2')
            selected_r2 = selected_scores.mean()
            selected_std = selected_scores.std()
            
            performance_comparison[model_name] = {
                'full_features_r2': full_r2,
                'full_features_std': full_std,
                'selected_features_r2': selected_r2,
                'selected_features_std': selected_std,
                'improvement': selected_r2 - full_r2
            }
            
            print(f"    全部特征: R² = {full_r2:.4f} (±{full_std:.4f})")
            print(f"    选择特征: R² = {selected_r2:.4f} (±{selected_std:.4f})")
            print(f"    性能变化: {selected_r2 - full_r2:+.4f}")
        
        performance_results[dist] = performance_comparison
        
        # 保存性能比较结果
        performance_df = pd.DataFrame([
            {
                'model': model_name,
                'full_features_r2': perf['full_features_r2'],
                'full_features_std': perf['full_features_std'],
                'selected_features_r2': perf['selected_features_r2'],
                'selected_features_std': perf['selected_features_std'],
                'improvement': perf['improvement']
            }
            for model_name, perf in performance_comparison.items()
        ])
        
        filename_perf = f'model_performance_comparison_{dist}.csv'
        performance_df.to_csv(filename_perf, index=False, encoding='utf-8-sig')
        print(f"  性能比较结果已保存到: {filename_perf}")
    
    return performance_results

def analyze_band_consistency_across_distances(datasets, importance_results):
    """
    分析不同距离下波段重要性的一致性
    识别与Mad相关的稳定重要波段
    """
    print("\n=== 跨距离波段重要性一致性分析 ===")
    
    # 1. 收集所有距离下的特征重要性
    all_importance_data = {}
    for dist, result in importance_results.items():
        importance_df = result['importance_df'].copy()
        # 只保留前50个最重要的特征
        top_50 = importance_df.head(50)[['feature', 'combined_score']].copy()
        top_50.columns = ['feature', f'{dist}_score']
        all_importance_data[dist] = top_50
    
    # 2. 合并所有距离的重要性数据
    merged_importance = all_importance_data['5mm'].copy()
    for dist in ['10mm', '15mm', '20mm']:
        merged_importance = merged_importance.merge(
            all_importance_data[dist][['feature', f'{dist}_score']], 
            on='feature', how='outer'
        )
    
    # 填充缺失值
    merged_importance = merged_importance.fillna(0)
    
    # 3. 计算跨距离一致性指标
    distance_columns = ['5mm_score', '10mm_score', '15mm_score', '20mm_score']
    
    # 计算每个特征在不同距离下的稳定性
    merged_importance['mean_score'] = merged_importance[distance_columns].mean(axis=1)
    merged_importance['std_score'] = merged_importance[distance_columns].std(axis=1)
    merged_importance['cv_score'] = merged_importance['std_score'] / (merged_importance['mean_score'] + 1e-8)  # 变异系数
    merged_importance['stability_score'] = 1 / (1 + merged_importance['cv_score'])  # 稳定性得分
    
    # 4. 识别稳定的重要波段
    # 条件：平均得分高 + 稳定性好
    stable_threshold = merged_importance['stability_score'].quantile(0.7)  # 稳定性前30%
    importance_threshold = merged_importance['mean_score'].quantile(0.8)   # 重要性前20%
    
    stable_important_bands = merged_importance[
        (merged_importance['stability_score'] >= stable_threshold) &
        (merged_importance['mean_score'] >= importance_threshold)
    ].copy()
    
    # 按综合得分排序
    stable_important_bands = stable_important_bands.sort_values('mean_score', ascending=False)
    
    # 5. 输出分析结果
    print(f"\n跨距离一致性分析结果:")
    print(f"  总特征数: {len(merged_importance)}")
    print(f"  稳定重要波段数: {len(stable_important_bands)}")
    print(f"  稳定性阈值: {stable_threshold:.4f}")
    print(f"  重要性阈值: {importance_threshold:.4f}")
    
    print(f"\n前20个最稳定的重要波段:")
    for i, row in stable_important_bands.head(20).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: 平均得分={row['mean_score']:.4f}, 稳定性={row['stability_score']:.4f}")
        print(f"      各距离得分: 5mm={row['5mm_score']:.4f}, 10mm={row['10mm_score']:.4f}, 15mm={row['15mm_score']:.4f}, 20mm={row['20mm_score']:.4f}")
    
    # 6. 保存结果
    merged_importance.to_csv('cross_distance_band_consistency.csv', index=False, encoding='utf-8-sig')
    stable_important_bands.to_csv('stable_important_bands.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n完整一致性分析结果已保存到: cross_distance_band_consistency.csv")
    print(f"稳定重要波段已保存到: stable_important_bands.csv")
    
    return stable_important_bands

def identify_physical_band_ranges(stable_bands):
    """
    识别物理上合理的波段范围
    基于光谱学原理，灰分相关波段应该在特定范围内
    """
    print("\n=== 物理波段范围分析 ===")
    
    # 提取波段数值
    band_values = []
    for feature in stable_bands['feature']:
        try:
            # 尝试提取数值
            if isinstance(feature, (int, float)):
                band_values.append(float(feature))
            elif isinstance(feature, str):
                # 如果是字符串，尝试提取数字
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
        
        # 识别主要波段群
        print(f"\n  主要波段群:")
        for i, band in enumerate(band_values):
            if i == 0 or band - band_values[i-1] > 10:  # 间隔大于10nm认为是新群
                print(f"    群 {i+1}: {band:.1f} nm")
            else:
                print(f"          {band:.1f} nm")
    
    return band_values

def recommend_band_selection_strategy(stable_bands, band_values):
    """
    基于分析结果推荐波段选择策略
    """
    print("\n=== 波段选择策略推荐 ===")
    
    print("1. 稳定性优先策略:")
    print("   选择稳定性得分最高的波段，确保在不同距离下都重要")
    
    print("\n2. 物理合理性策略:")
    print("   基于光谱学原理，重点关注以下波段范围:")
    print("   - 900-1000 nm: C-H键的第三倍频")
    print("   - 1100-1200 nm: C-H键的第二倍频")
    print("   - 1400-1500 nm: O-H键的第一倍频")
    
    print("\n3. 实际应用建议:")
    print("   - 选择稳定性得分 > 0.8 的波段")
    print("   - 优先选择在多个距离下都表现良好的波段")
    print("   - 考虑波段之间的相关性，避免冗余")
    
    # 推荐具体波段
    top_stable = stable_bands.head(10)
    print(f"\n4. 推荐的前10个稳定波段:")
    for i, row in top_stable.iterrows():
        print(f"   {i+1:2d}. {row['feature']} (稳定性: {row['stability_score']:.3f})")
    
    return top_stable