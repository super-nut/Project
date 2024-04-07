# -*- coding: utf-8 -*-
"""
@author: JHL
"""

"""
---- 01-数据清洗 ----
"""
import pandas as pd

# 1. 读取文件
df = pd.read_excel("dqacc数据集-00-预处理前.xlsx", header=None)

# 2. 删除含有大于等于一定比例的空值的行
total_feature_count = len(df.columns[:-1])  # 总特征数（第一列到倒数第二列）
threshold = 0.8 * total_feature_count  # 删除80%以上空值的行
df.dropna(thresh=threshold, inplace=True)

# 3. 缺失值填充
# 这里假设数值型特征使用均值填充，分类特征使用众数填充
numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(include='object').columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# 4. 删除离群值点所在行（仅针对数值型特征）
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    threshold = 3 * std  # 3σ原则

    df = df[(df[col] >= mean - threshold) & (df[col] <= mean + threshold)]

# 保存处理后的数据集
df.to_excel("dqacc数据集-01-数据清洗.xlsx", index=False)

"""
---- 02-不平衡数据处理 ----
"""
from imblearn.over_sampling import SMOTE


def process_smote():
    # smote过采样
    data = pd.read_excel('dqacc数据集-01-数据清洗.xlsx')

    features = data.iloc[1:, :-1]  # 第1列到倒数第二列为特征列
    labels = data.iloc[1:, -1]  # 最后一列为标签列

    # 使用SMOTE算法进行过采样
    oversample = SMOTE()
    os_features, os_labels = oversample.fit_resample(features, labels)

    # 将过采样后的数据转换为DataFrame
    oversampled_data = pd.DataFrame(os_features, columns=features.columns)
    oversampled_data['label'] = os_labels

    shuffled_data = oversampled_data.sample(frac=1).reset_index(drop=True)

    # 可以将oversampled_data保存为新的Excel文件
    shuffled_data.to_excel('dqacc数据集-02-不平衡数据处理.xlsx', index=False)


# 调用函数进行处理
process_smote()

"""
---- 03-标准化处理 ----
"""
from sklearn.preprocessing import MinMaxScaler


def normalize_data():
    # 读取处理后的数据
    data = pd.read_excel('dqacc数据集-02-不平衡数据处理.xlsx', header=None)  # 由于数据没有列名，需要设置header=None

    # 提取特征和标签列
    features = data.iloc[1:, 2:-1]  # 第二行到最后一行，第三列到倒数第二列为特征列
    labels = data.iloc[1:, -1]  # 最后一列为标签列

    # 使用 Min-Max 标准化对特征进行规范化处理
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # 构建规范化后的DataFrame
    normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_data['label'] = labels  # 添加标签列

    # 保存规范化后的数据
    normalized_data.to_excel('dqacc数据集-03-标准化处理.xlsx', index=False)


normalize_data()

"""
---- 04-特征筛选 ----
"""

import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

# 1. 读取文件，并指定dtype='str'
df = pd.read_excel("dqacc数据集-02-不平衡数据处理.xlsx", header=None, dtype='str')

# 2. 对非数值类型的列进行 Label Encoding
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

X = df.iloc[1:, :-1]  # 特征从第一列到倒数第二列，数据从第二行开始
y = df.iloc[1:, -1]  # 标签在最后一列

# 3. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 3. 参数调优
def adjust_params(X_train, y_train):
    xgb_cls = XGBClassifier(random_state=1, n_jobs=-1, objective='binary:logistic',
                            eval_metric='auc')  # ,**other_params
    params_set = {'n_estimators': [5, 10, 15, 20], "learning_rate": [0.1, 0.001, 0.01],
                  'max_depth': [i for i in range(3, 11)]}  # depth of tree

    grid = GridSearchCV(xgb_cls, cv=5, param_grid=params_set)
    grid.fit(X_train, y_train)
    other_params = grid.best_params_
    print("best params:", other_params)
    return other_params


other_params = adjust_params(X_train, y_train)
xgb_cls = XGBClassifier(random_state=1, n_jobs=-1, objective='binary:logistic', eval_metric='auc', **other_params)
xgb_cls.fit(X_train, y_train)

# 4. 测试模型
y_pred = xgb_cls.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy: %.2f%%" % (acc * 100.0))

# 5. 特征重要度评估
plt.figure(figsize=(12, 8))  # 假设要生成 12x8 英寸的图像
Available_importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
for importance_type in Available_importance_types:
    print('%s:' % importance_type, xgb_cls.get_booster().get_score(importance_type=importance_type))
    plot_importance(xgb_cls)
    # plt.show()
plt.savefig('figure.png', bbox_inches='tight')  # bbox_inches='tight'用于确保图像周围没有空白边缘

# 6. 递归特征消除-交叉验证
estimator = xgb_cls
selector = RFECV(estimator=estimator, cv=3)
selector.fit(X_train, y_train)
print("number of features %s" % selector.n_features_)
print("support is %s" % selector.support_)
print("ranking of features is%s" % selector.ranking_)
print("grid scores %s" % selector.cv_results_)


if __name__ == '__main__':
    pass