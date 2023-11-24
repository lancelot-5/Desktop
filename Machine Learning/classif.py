
# pip install imbalanced-learn
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
#欠采样
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek

class SMOTE:
    def __init__(self, samples, N=10, k=5):
        self.n_samples, self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic
    # for each minority class samples,
    # choose N of the k nearest neighbors and
    # generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1


# 1.获取数据集
dataset = pd.read_csv("yeast3.dat", header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# 2.划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# 3.调用SMOTE过采样算法对少数类样本进行过采样
idx = np.where(y_train == 1)[0]

# N:过采样数量; k:近邻数量

kos = SMOTETomek(random_state=0) # 综合采样

X_train_oversampled, y_train_oversampled = kos.fit_resample(X, y)


# 4.训练分类器
# 调用支持向量机分类器
model1 = SVC()
model_oversampled1 = SVC()
model2 = tree.DecisionTreeClassifier()
model_oversampled2 = tree.DecisionTreeClassifier()
model3 = tree.DecisionTreeClassifier()
model_oversampled3 = tree.DecisionTreeClassifier()


clf1 = model1.fit(X_train, y_train)
clf_oversampled1= model_oversampled1.fit(X_train_oversampled, y_train_oversampled)

clf2 = model2.fit(X_train, y_train)
clf_oversampled2 = model_oversampled2.fit(X_train_oversampled, y_train_oversampled)

clf3 = model3.fit(X_train, y_train)
clf_oversampled3 = model_oversampled3.fit(X_train_oversampled, y_train_oversampled)
# 5.获取预测结果
pred1 = clf1.predict(X_test)
pred_oversampled1 = clf_oversampled1.predict(X_test)
pred2 = clf2.predict(X_test)
pred_oversampled2 = clf_oversampled2.predict(X_test)
pred3 = clf3.predict(X_test)
pred_oversampled3 = clf_oversampled3.predict(X_test)
# 6.使用ACU和F1-score评估预测结果
print("使用SVM AUC/F1 for original data: ", roc_auc_score(y_test, pred1), f1_score(y_test, pred1))
print("使用SVM AUC/F1 for oversampled data: ", roc_auc_score(y_test, pred_oversampled1), f1_score(y_test, pred_oversampled1))
print("使用SVM 准确率+召回率 for original data: ", accuracy_score(y_test, pred1),recall_score(y_test, pred1))
print("使用SVM 准确率+召回率 for oversampled data: ", accuracy_score(y_test, pred_oversampled1),recall_score(y_test, pred_oversampled1))
print("*******************************************************************")
print("使用决策树 AUC/F1 for original data: ", roc_auc_score(y_test, pred2), f1_score(y_test, pred2))
print("使用决策树 AUC/F1 for oversampled data: ", roc_auc_score(y_test, pred_oversampled2), f1_score(y_test, pred_oversampled2))
print("使用决策树 准确率+召回率 for original data: ", accuracy_score(y_test, pred2),recall_score(y_test, pred2))
print("使用决策树 准确率+召回率 for oversampled data: ", accuracy_score(y_test, pred_oversampled2),recall_score(y_test, pred_oversampled2))
print("*******************************************************************")
print("使用贝叶斯 AUC/F1 for original data: ", roc_auc_score(y_test, pred3), f1_score(y_test, pred3))
print("使用贝叶斯 AUC/F1 for oversampled data: ", roc_auc_score(y_test, pred_oversampled3), f1_score(y_test, pred_oversampled3))
print("使用贝叶斯 准确率+召回率 for original data: ", accuracy_score(y_test, pred3),recall_score(y_test, pred3))
print("使用贝叶斯 准确率+召回率 for oversampled data: ", accuracy_score(y_test, pred_oversampled3),recall_score(y_test, pred_oversampled3))