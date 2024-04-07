# -*- coding: utf-8 -*-
"""
@author: JHL
"""

"""
----- 数据读取和预处理 -----
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  #分割数据集

data = pd.read_excel('dqacc数据集-02-不平衡数据处理.xlsx', header=0)
data = data.to_numpy()

#列名变为第一行
features = data[:, :-1].astype('float32')
labels = data[:, -1].astype('int32')
print(f"Dataset shape:\nfeatures {features.shape}  labels {labels.shape}")

# 使用 Min-Max 标准化对特征进行规范化处理
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

#简单划分训练测试集
test_ratio = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(normalized_features, labels, test_size=test_ratio, random_state=42)
print(f"test_ratio: {test_ratio}")


"""
----- 评价指标计算 -----
"""
def precision(output, target):
    confusion_matrix = [[0, 0], [0, 0]]
    for t, p in zip(target, output):
        confusion_matrix[t.astype(int)][p.astype(int)] += 1

    return confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1] + 1e-6)


def recall(output, target):
    confusion_matrix = [[0, 0], [0, 0]]
    for t, p in zip(target, output):
        confusion_matrix[t.astype(int)][p.astype(int)] += 1

    return confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1] + 1e-6)


def f1(output, target):
    p = precision(output, target)
    r = recall(output, target)
    return 2 * p * r / (p + r + 1e-6)


"""
----- 随机森林算法 -----
"""
def fun1():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42, min_samples_split=3,n_estimators=500,min_samples_leaf=4,
                                        min_impurity_decrease=1e-8,n_jobs=-1,max_features=0.7,max_depth=20)

    model.fit(X_train,Y_train)

    Y_test_pre = model.predict(X_test)

    acc=sum(Y_test_pre==Y_test)/len(Y_test)
    p = precision(Y_test_pre, Y_test)
    r = recall(Y_test_pre, Y_test)
    f = f1(Y_test_pre, Y_test)
    print(f'Accuracy：{acc}\n precision:{p}\n recall:{r}\n f1:{f}')


"""
----- 决策树算法 -----
"""
def fun2():
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(criterion='entropy', splitter="best", max_depth=74, min_samples_leaf=2,min_samples_split=2, random_state=22)

    model.fit(X_train,Y_train)

    Y_test_pre = model.predict(X_test)

    acc=sum(Y_test_pre==Y_test)/len(Y_test)
    p = precision(Y_test_pre, Y_test)
    r = recall(Y_test_pre, Y_test)
    f = f1(Y_test_pre, Y_test)
    print(f'Accuracy：{acc}\n precision:{p}\n recall:{r}\n f1:{f}')


"""
----- 支持向量机 -----
"""
def fun3():
    import sklearn

    model = sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='auto', coef0=0.0, shrinking=True,
                    probability=True, tol=0.0001, max_iter=-1, random_state=42)

    model.fit(X_train,Y_train)

    Y_test_pre = model.predict(X_test)

    acc=sum(Y_test_pre==Y_test)/len(Y_test)
    p = precision(Y_test_pre, Y_test)
    r = recall(Y_test_pre, Y_test)
    f = f1(Y_test_pre, Y_test)
    print(f'Accuracy：{acc}\n precision:{p}\n recall:{r}\n f1:{f}')


"""
----- 朴素贝叶斯算法(多项式) -----
"""
def fun4():
    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB(class_prior=[0.5, 0.5])

    model.fit(X_train,Y_train)

    Y_test_pre = model.predict(X_test)

    acc=sum(Y_test_pre==Y_test)/len(Y_test)
    p = precision(Y_test_pre, Y_test)
    r = recall(Y_test_pre, Y_test)
    f = f1(Y_test_pre, Y_test)
    print(f'Accuracy：{acc}\n precision:{p}\n recall:{r}\n f1:{f}')


if __name__ == '__main__':
    fun1()
    # fun2()
    # fun3()
    # fun4()