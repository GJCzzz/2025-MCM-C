import pandas as pd
import numpy as np
from pmdarima import auto_arima

# 读取数据
df = pd.read_csv("summerOly_score.csv")

# 计算得分率
df["Score_Rate"] = df["Total Score"] / df["Yearly Total Score"]

# 按国家和年份排序
df = df.sort_values(["NOC", "Year"]).reset_index(drop=True)


# 定义自动建模函数
def calculate_lag_impact(country_df):
    """为单个国家计算滞后项影响"""
    country_df = country_df.sort_values("Year")
    max_lags = min(3, len(country_df) - 1)  # 动态确定最大可用滞后阶数

    # 如果 max_lags < 1，说明数据不足，直接返回 None
    if max_lags < 1:
        return None

    # 生成滞后特征
    for lag in range(1, max_lags + 1):
        country_df[f"lag{lag}"] = country_df["Score_Rate"].shift(lag)

    # 删除包含缺失值的行（无法计算滞后项的情况）
    country_df = country_df.dropna(subset=[f"lag{max_lags}"], how="any")
    if len(country_df) < 2:  # 至少需要2个数据点建模
        return None

    # 准备数据
    endog = country_df["Score_Rate"]
    exog = country_df[[f"lag{i}" for i in range(1, max_lags + 1)]]

    # 自动建模
    try:
        model = auto_arima(
            y=endog,
            X=exog,
            seasonal=False,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True
        )
    except:
        return None

    # 提取系数
    coeffs = {}
    for i, lag in enumerate(range(1, max_lags + 1)):
        coeff_name = f"lag{lag}"
        coeff_value = model.params().get(coeff_name, np.nan)
        coeffs[f"lag{lag}_coeff"] = coeff_value
        coeffs[f"lag{lag}_pvalue"] = model.pvalues().get(coeff_name, np.nan)

    return pd.Series({
        "Obs_Years": len(country_df),
        **coeffs
    })


# 对每个国家进行分析
results = []
for noc, group in df.groupby("NOC"):
    res = calculate_lag_impact(group)
    if res is not None:
        res["NOC"] = noc
        results.append(res)

# 转换为DataFrame并保存
result_df = pd.DataFrame(results).set_index("NOC")
result_df.to_csv("olympic_lag_impact.csv")

print("分析完成，结果已保存到 olympic_lag_impact.csv")
print("\n示例输出：")
print(result_df.head(10))