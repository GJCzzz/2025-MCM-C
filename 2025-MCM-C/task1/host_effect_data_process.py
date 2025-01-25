import pandas as pd

# 读取奖牌数据
medal_df = pd.read_csv(r'D:\AAAAhust\数学建模\2025MCM\2025-MCM-C\2025_Problem_C_Data\summerOly_medal_counts(1).csv')

# 读取东道主信息
host_data = {
    'Year': [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'Host Country': ['Greece', 'France', 'United States', 'Great Britain', 'Sweden', 'Belgium', 'France', 'Netherlands', 'United States', 'Germany', 'Great Britain', 'Finland', 'Australia', 'Italy', 'Japan', 'Mexico', 'West Germany', 'Canada', 'Soviet Union', 'United States', 'South Korea', 'Spain', 'United States', 'Australia', 'Greece', 'China', 'Great Britain', 'Brazil', 'Japan', 'France']
}
host_df = pd.DataFrame(host_data)

# 计算每个国家的总分
medal_df['Total Score'] = medal_df['Gold'] * 3 + medal_df['Silver'] * 2 + medal_df['Bronze'] * 1

# 合并东道主信息
medal_df['Is Host'] = 0
for index, row in host_df.iterrows():
    medal_df.loc[(medal_df['Year'] == row['Year']) & (medal_df['NOC'] == row['Host Country']), 'Is Host'] = 1

# 计算每一届奥运会的总得分
yearly_total = medal_df.groupby('Year')['Total Score'].sum().reset_index()
yearly_total.rename(columns={'Total Score': 'Yearly Total Score'}, inplace=True)
medal_df = pd.merge(medal_df, yearly_total, on='Year')

# 保存结果到新的CSV文件
medal_df.to_csv('summerOly_score.csv', index=False)

# 打印结果
print(medal_df)