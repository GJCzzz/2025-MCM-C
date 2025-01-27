import pandas as pd
from linearmodels import PanelOLS

# 读取数据
df = pd.read_csv("performance(1).csv")

# 处理缺失值和数据类型
df["Performance_Spike"] = df["Performance_Spike"].replace({"FALSE": 0, "TRUE": 1}).astype(int)
df["Host"] = df["Host"].astype(int)
df["EventCount"] = df["EventCount"].fillna(0)  # 填充项目数量缺失值
df["PersonCount"] = df["PersonCount"].fillna(0)  # 填充参赛人数缺失值

# 设置多级索引（国家-年份）
df = df.set_index(["NOC", "Year"])

# 检查 Performance_Spike 是否被完全吸收
spike_counts = df["Performance_Spike"].value_counts()
if len(spike_counts) == 1:  # 如果全为 0 或全为 1
    raise ValueError("Performance_Spike 被完全吸收，无法建模")

# 定义自变量
exog_vars = ["Performance_Spike", "Host", "PersonCount"]

# 定义回归模型
model = PanelOLS(
    dependent=df["Total_Score"],  # 因变量：总得分
    exog=df[exog_vars],  # 自变量
    entity_effects=True,  # 控制国家-年份的固定效应
    drop_absorbed=True  # 自动移除被吸收的变量
)

# 拟合模型并输出结果
results = model.fit(cov_type="clustered", cluster_entity=True)
print(results.summary)

# 提取绩效突变的系数及其置信区间
beta = results.params["Performance_Spike"]
ci_low = results.conf_int().loc["Performance_Spike", "lower"]
ci_high = results.conf_int().loc["Performance_Spike", "upper"]

print(f"\n绩效突变的全局贡献度：")
print(f"- 系数: {beta:.2f}")
print(f"- 95% 置信区间: [{ci_low:.2f}, {ci_high:.2f}]")