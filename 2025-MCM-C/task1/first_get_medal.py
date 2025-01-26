import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# 读取数据
# 假设你已经有一个包含特征和标签的DataFrame
# 列名：['NOC', '平均每届参赛人次', '平均每届参赛项目数', '参加届数', '标签']
data = pd.read_csv('first_get_medal.csv',encoding='GBK')

# 特征和标签
X = data[['平均每届参赛人次', '平均每届参赛项目数', '参加届数']]
y = data['标签']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

# 绘制 ROC 曲线
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.show()

# 预测下一届奥运会可能首次取得奖牌的国家
# 假设有一个包含所有从未获得过奖牌的国家的DataFrame
# 列名：['NOC', '平均每届参赛人次', '平均每届参赛项目数', '参加届数']
never_won_medals = pd.read_csv('no_medal_countries_with_stats.csv',encoding='GBK')

# 提取特征
X_new = never_won_medals[['平均每届参赛人次', '平均每届参赛项目数', '参加届数']]

# 预测概率
never_won_medals['预测概率'] = model.predict_proba(X_new)[:, 1]

# 按概率排序，找到最有可能首次取得奖牌的国家
most_likely_countries = never_won_medals.sort_values(by='预测概率', ascending=False)
print("最有可能首次取得奖牌的国家:\n", most_likely_countries[['NOC', '预测概率']])

# 保存预测结果
most_likely_countries.to_csv('most_likely_countries.csv', index=False)
print("预测结果已保存到 most_likely_countries.csv 文件中。")