import pandas as pd
import numpy as np

# 读取CSV文件
file_path = 'olympic_lag_impact.csv'
df = pd.read_csv(file_path)

# 定义需要归一化的列
columns_to_normalize = ['lag1_coeff', 'lag2_coeff', 'lag3_coeff']

# Decimal Scaling 归一化函数
def decimal_scaling(column):
    max_abs_value = np.max(np.abs(column))  # 找到列中绝对值的最大值
    j = np.ceil(np.log10(max_abs_value))    # 计算基数 10^j
    return column / (10 ** j)               # 应用 Decimal Scaling

# 对指定列进行 Decimal Scaling 归一化
for col in columns_to_normalize:
    df[col] = decimal_scaling(df[col])

# 只保留指定的3列数据
df_normalized = df[columns_to_normalize]

# 保存处理后的数据到新文件
output_file_path = 'olympic_lag_impact_decimal_scaled.csv'
df_normalized.to_csv(output_file_path, index=False)

print(f"Decimal Scaling归一化后的数据已保存到: {output_file_path}")
