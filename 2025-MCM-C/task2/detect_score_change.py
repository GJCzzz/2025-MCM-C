import pandas as pd
import numpy as np

# 读取处理后的数据
df = pd.read_csv('country_sport_year_summary_1.csv')


# 定义一个函数，检测某个国家-运动组合的成绩突变
def detect_performance_spike(data, country, sport, window=5, threshold_multiplier=1.5, min_score_increase=5,
                             min_periods=2):
    """
    检测某个国家-运动组合的成绩突变。

    参数：
    - data: 包含国家、运动、年份、总得分的数据框。
    - country: 目标国家（NOC）。
    - sport: 目标运动。
    - window: 滚动窗口大小（默认5届奥运会）。
    - threshold_multiplier: 阈值倍数（默认1.5倍标准差）。
    - min_score_increase: 最小得分增加量（默认5分）。
    - min_periods: 最小届数要求（默认2届）。

    返回：
    - 包含年份和是否突变的标记的数据框。
    """
    # 筛选目标国家-运动组合的数据
    target_data = data[(data['NOC'] == country) & (data['Sport'] == sport)].copy()
    target_data = target_data.sort_values('Year')  # 按年份排序

    # 如果数据不足最小届数要求，返回空结果
    if len(target_data) < min_periods:
        return pd.DataFrame()

    # 计算滚动均值和标准差，动态调整窗口大小
    target_data['Rolling_Mean'] = (
        target_data['Total_Score']
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )
    target_data['Rolling_Std'] = (
        target_data['Total_Score']
        .rolling(window=window, min_periods=min_periods)
        .std()
    )

    # 处理标准差为0的情况
    target_data['Rolling_Std'] = target_data['Rolling_Std'].fillna(0)  # 填充NaN为0
    target_data['Threshold'] = (
            target_data['Rolling_Mean'] + threshold_multiplier * target_data['Rolling_Std']
    )

    # 检测突变：超过阈值或得分增加超过最小增量
    target_data['Performance_Spike'] = (
            (target_data['Total_Score'] > target_data['Threshold']) |  # 相对阈值
            (target_data['Total_Score'] - target_data['Rolling_Mean'] > min_score_increase)  # 绝对阈值
    )

    return target_data[['Year', 'Total_Score', 'Rolling_Mean', 'Rolling_Std', 'Threshold', 'Performance_Spike']]


# 示例：检测阿富汗（AFG）在田径（Athletics）上的成绩突变
afg_athletics_spikes = detect_performance_spike(df, country='AFG', sport='Athletics')
print(afg_athletics_spikes)


# 批量检测所有国家-运动组合
def detect_spikes_for_all(data, window=5, threshold_multiplier=1.5, min_score_increase=5, min_periods=2):
    """
    检测所有国家-运动组合的成绩突变。

    参数：
    - data: 包含国家、运动、年份、总得分的数据框。
    - window: 滚动窗口大小（默认5届奥运会）。
    - threshold_multiplier: 阈值倍数（默认1.5倍标准差）。
    - min_score_increase: 最小得分增加量（默认5分）。
    - min_periods: 最小届数要求（默认2届）。

    返回：
    - 包含所有检测结果的数据框。
    """
    results = []
    grouped = data.groupby(['NOC', 'Sport'])

    for (country, sport), group in grouped:
        spikes = detect_performance_spike(data, country, sport, window, threshold_multiplier, min_score_increase,
                                          min_periods)
        if not spikes.empty:
            spikes['NOC'] = country
            spikes['Sport'] = sport
            results.append(spikes)

    return pd.concat(results)


# 批量检测所有国家-运动组合
all_spikes = detect_spikes_for_all(df)
all_spikes.to_csv('performance_spikes_2.csv', index=False)  # 保存结果
print(all_spikes.head())