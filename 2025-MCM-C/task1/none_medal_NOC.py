import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'D:\AAAAhust\数学建模\2025-mcm\2025-MCM-C\2025_Problem_C_Data\summerOly_athletes(1).csv')

# 删除1906年的数据
df = df[df['Year'] != 1906]

# 筛选出从未获得奖牌的国家
# 首先找出所有取得过奖牌的国家
medal_countries = df[df['Medal'] > 0]['NOC'].unique()

# 然后找出所有从未取得过奖牌的国家
no_medal_countries = df[~df['NOC'].isin(medal_countries)]['NOC'].unique()

# 将结果保存为DataFrame
no_medal_df = pd.DataFrame(no_medal_countries, columns=['NOC'])

# 计算每个国家实际参加的奥运会届数
participation_data = df.groupby('NOC')['Year'].nunique().reset_index()
participation_data.columns = ['NOC', '参加届数']

# 计算每个国家的总参赛人次
total_participations = df.groupby('NOC').size().reset_index(name='总参赛人次')

# 计算每个国家的总参赛项目数
total_events = df.groupby(['NOC', 'Event']).size().reset_index().groupby('NOC').size().reset_index(name='总参赛项目数')

# 合并数据
no_medal_with_stats = pd.merge(no_medal_df, participation_data, on='NOC', how='left')
no_medal_with_stats = pd.merge(no_medal_with_stats, total_participations, on='NOC', how='left')
no_medal_with_stats = pd.merge(no_medal_with_stats, total_events, on='NOC', how='left')

# 计算平均每届参赛人次
no_medal_with_stats['平均每届参赛人次'] = no_medal_with_stats['总参赛人次'] / no_medal_with_stats['参加届数']

# 计算平均每届参赛项目数
no_medal_with_stats['平均每届参赛项目数'] = no_medal_with_stats['总参赛项目数'] / no_medal_with_stats['参加届数']

# 保存结果到CSV文件
no_medal_with_stats.to_csv('no_medal_countries_with_stats.csv', index=False)

print("从未获得奖牌的国家及其参赛统计数据已保存到 no_medal_countries_with_stats.csv 文件中。")