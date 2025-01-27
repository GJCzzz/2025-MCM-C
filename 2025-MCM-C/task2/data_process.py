import pandas as pd

# 读取数据
df = pd.read_csv('athlete.csv')

# 确保列名正确
# 假设列名如下：
# Name, Sex, Team, NOC, Host, Year, City, Sport, Event, Medal

# 处理缺失值：将 Medal 列中的缺失值填充为 0
df['Medal'] = df['Medal'].fillna(0)

# 按国家、运动、年份分组，计算参赛人数（去重）和总得分
grouped = df.groupby(['NOC', 'Sport', 'Year']).agg(
    Athlete_Count=('Name', lambda x: x.nunique()),  # 参赛人数（去重）
    Total_Score=('Medal', 'sum')                   # 总得分
).reset_index()

# 保存结果到新的CSV文件
grouped.to_csv('country_sport_year_summary_1.csv', index=False)

# 打印前几行查看结果
print(grouped.head())