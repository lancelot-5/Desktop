from skmultilearn.dataset import load_from_arff
from sklearn.metrics import roc_auc_score, f1_score , accuracy_score,recall_score
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
def MLSMOTE(X,y, n_sample):
    """Give the augmented data using MLSMOTE algorithm
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data"""
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0,n-1)
        neighbour = random.choice(indices2[reference,1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbour,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target
def get_minority_instace(X, y):
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub
def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    args
    X: np.array, array whose nearest neighbor has to find
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
    euclidean,indices= nbs.kneighbors(X)
    return indices
def get_index(df):
    """
    give the index of all tail_label rows
    args
    df: pandas.DataFrame, target label df from which index for tail label has to identified
    return
    index: list, a list containing index number of all the tail label
    """
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label]==1].index)
        index = index.union(sub_index)
    return list(index)
def get_tail_label(df):
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label
#import mlsmote
label_count_datasheet = {'emotions': 6, 'yeast': 14, 'bibtex': 159}
# 1.获取数据集(训练集，测试集)
X_train, y_train = load_from_arff(r"emotions/emotions-train.arff", label_count_datasheet['emotions'])
X_test, y_test = load_from_arff(r"emotions/emotions-test.arff", label_count_datasheet['emotions'])
# 2.调用MLSMOTE过采样算法进行过采样(https://github.com/niteshsukhwani/MLSMOTE)
X_train = pd.DataFrame(X_train.toarray())
y_train = pd.DataFrame(y_train.toarray())
X_sub, y_sub = get_minority_instace(X_train, y_train)
# 得到过采样后的数据
X_resampled_mlsmote, y_resampled_mlsmote = MLSMOTE(X_sub, y_sub, 30)
X_resampled_mlsmote = X_resampled_mlsmote.append(X_train)
y_resampled_mlsmote = y_resampled_mlsmote.append(y_train)
# 3.训练BR分类器（基分类器使用贝叶斯）


clf = ClassifierChain(GaussianNB())
clf_oversampled = ClassifierChain(GaussianNB())
# clf = LabelPowerset(classifier=SVC())
# clf_oversampled = LabelPowerset(classifier=SVC())
# 4.获取预测结果
pred = clf.fit(X_train, y_train).predict(X_test)
pred_oversampled = clf_oversampled.fit(X_resampled_mlsmote, y_resampled_mlsmote).predict(X_test)
# 5.使用Macro ACU和Macro F1-score评估预测结果
print("使用贝叶斯得到的Macro AUC/F1 for original data: ", roc_auc_score(y_test.toarray(), pred.toarray(), average="macro"),
f1_score(y_test, pred, average="macro"))
print(accuracy_score(y_test, pred.toarray()),recall_score(y_test, pred.toarray(),average="macro"))
print(accuracy_score(y_test, pred_oversampled.toarray()),recall_score(y_test, pred_oversampled.toarray(),average="macro"))

print("Macro AUC/F1 for oversampled data: ", roc_auc_score(y_test.toarray(), pred_oversampled.toarray(), average="macro"),
f1_score(y_test, pred_oversampled, average="macro"))