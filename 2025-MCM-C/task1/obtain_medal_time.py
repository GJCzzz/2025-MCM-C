import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'D:\AAAAhust\数学建模\2025-mcm\2025-MCM-C\2025_Problem_C_Data\summerOly_athletes(1).csv')

# 删除1906年的数据
df = df[df['Year'] != 1906]

# 按国家和年份分组，统计每届奥运会是否获得奖牌
# 如果某届奥运会中至少有一块奖牌（成绩 > 0），则标记为 True
medal_data = df.groupby(['NOC', 'Year'])['Medal'].apply(lambda x: (x > 0).any()).reset_index()
medal_data.columns = ['NOC', 'Year', '是否获得奖牌']

# 找到每个国家首次获得奖牌的年份
first_medal = medal_data[medal_data['是否获得奖牌']].groupby('NOC')['Year'].min().reset_index()
first_medal.columns = ['NOC', '首次获得奖牌年份']

# 找到每个国家首次参加奥运会的年份
first_participation = df.groupby('NOC')['Year'].min().reset_index()
first_participation.columns = ['NOC', '首次参加年份']

# 找到每个国家实际参加的奥运会届数
# 按国家分组，统计每届奥运会的参加情况
participation_data = df.groupby(['NOC', 'Year']).size().reset_index()
participation_data = participation_data[['NOC', 'Year']]

# 为每个国家计算参加奥运会的届数
participation_data['参加届数'] = participation_data.groupby('NOC').cumcount() + 1

# 合并首次参加年份和首次获得奖牌年份
result = pd.merge(first_participation, first_medal, on='NOC', how='left')

# 找到每个国家首次获得奖牌时的参加届数
result = pd.merge(result, participation_data, left_on=['NOC', '首次获得奖牌年份'], right_on=['NOC', 'Year'], how='left')
result = result[['NOC', '首次参加年份', '首次获得奖牌年份', '参加届数']]

# 处理从未获得奖牌的国家
result['参加届数'] = result['参加届数'].fillna(-1)  # -1 表示从未获得奖牌

# 保存结果到CSV文件
result.to_csv('first_medal_analysis_1.csv', index=False)

print("分析结果已保存到 first_medal_analysis_1.csv 文件中。")