import pandas as pd
import statsmodels.api as sm

# 1. 加载数据
# 假设你的数据文件是 summerOly_score.csv
df = pd.read_csv('summerOly_score.csv')

# 2. 数据预处理
# 确保数据列名与文件内容一致
# 如果列名有空格或特殊字符，可以重命名
df.columns = ['Rank', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Year', 'Total Score', 'Is Host', 'Yearly Total Score']

# 计算得分率
df['ScoreRate'] = df['Total Score'] / df['Yearly Total Score']

# 创建虚拟变量：是否为主办国
df['IsHost'] = df['Is Host'].astype(int)

# 添加常数项
df['Intercept'] = 1

# 3. 定义自变量和因变量
# 自变量：IsHost（是否为主办国）、NOC（国家）、Year（届次）
# 因变量：ScoreRate（得分率）
X = df[['Intercept', 'IsHost', 'NOC', 'Year']]  # 包含 NOC 和 Year
y = df['ScoreRate']  # 因变量

# 添加固定效应（国家固定效应和届次固定效应）
X = pd.get_dummies(X, columns=['NOC', 'Year'], drop_first=True)

# 确保 X 和 y 是数值类型
X = X.astype(float)  # 将 X 转换为浮点类型
y = y.astype(float)  # 将 y 转换为浮点类型

# 4. 拟合回归模型
model = sm.OLS(y, X)
results = model.fit()

# 5. 输出回归结果
print(results.summary())

# 6. 提取东道主效应
host_effect = results.params['IsHost']
print(f"东道主效应（得分率）：{host_effect}")