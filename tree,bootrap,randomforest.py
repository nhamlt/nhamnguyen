# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:41:18 2022

@author: admin
"""

import pandas as pd
import numpy as np
import sklearn as sklean
import sklearn.linear_model as linnear_model
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as stats
import scipy
import statsmodels.api as sm
from sklearn import tree as tree
#%% load data

data = pd.read_csv('C:/Users/admin/Desktop/smarket.csv',header = 'infer',index_col=None)
df = data.copy()
Tree =tree.DecisionTreeClassifier(random_state=0,ccp_alpha=8.08626350e-05)
x = df.iloc[:,2:7]
y= df['Direction']
Xtrain,Xtest,ytrain,ytest = sklean.model_selection.train_test_split(x,y,test_size=0.2)
data = pd.read_csv('C:/Users/admin/Desktop/train.csv/train.csv',header = 'infer')
Xtrain,Xtest,ytrain,ytest = sklean.model_selection.train_test_split(x,y,test_size=0.2)
model = Tree.fit(Xtrain,ytrain)
model.ccp_alpha
ypred = model.predict(Xtest)
MSE_test = sklean.metrics.confusion_matrix(ytest,ypred)
np.sum(np.diag(MSE_test))
7206/8400
text = tree.export_text(model)
model.tree_.node_count
import sklearn.discriminant_analysis as DA
LDA1 = DA.LinearDiscriminantAnalysis(solver='svd') # svd neu nhieu chieu, lsqr neu it chieu,eigen la tu dong
model2 = LDA1.fit(Xtrain,ytrain)
ypred=model2.predict(Xtest)
sklean.metrics.confusion_matrix(ytest,ypred)#58% xam lol
MSE_test = sklean.metrics.confusion_matrix(ytest,ypred)
np.sum(np.diag(MSE_test))
RidgeCV = linnear_model.RidgeClassifierCV(alphas=(0.01,10,100),fit_intercept= True,cv = 10)
model = RidgeCV.fit(Xtrain,ytrain)
ypred=model.predict(Xtest)
sklean.metrics.confusion_matrix(ytest,ypred)#58% xam lol
MSE_test = sklean.metrics.confusion_matrix(ytest,ypred)
np.sum(np.diag(MSE_test))
MSE=[]
from sklearn import neighbors
for i in range(15):
 KNN = neighbors.KNeighborsClassifier(n_neighbors=i+2, weights='distance',p = 2)
 model_KNN = KNN.fit(Xtrain,ytrain)
 ypred  = model_KNN.predict(Xtest)
 MSE_test = sklean.metrics.confusion_matrix(ytest,ypred)
 a=np.sum(np.diag(MSE_test))
 print(i+2,a,a/8400)
#%% plot tree
tree.plot_tree(model,feature_names=Xtrain.columns.values.tolist())
#%% finding alpha meanning findding node off tree
path = model.cost_complexity_pruning_path(Xtrain, ytrain)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
a = np.where(ccp_alphas<0.00002976)
a= [21*i for i in range(100)]

ccpnew = ccp_alphas[a]
ccp_alphas

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
a = np.where(np.logical_and(ccp_alphas<0.000088,ccp_alphas>0.000080))
ccpnew = ccp_alphas[a]
ccpnew
len(ccpnew)
clfs = []
i = 1
for ccp_alpha in ccpnew:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(Xtrain, ytrain)
    clfs.append(clf)
    i=i+1
    print(i)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)
len(clfs)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]

fig, ax = plt.subplots(2, 1)
ax[0].plot(ccpnew, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccpnew, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(Xtrain, ytrain) for clf in clfs]
test_scores = [clf.score(Xtest, ytest) for clf in clfs]

best_alphas= pd.DataFrame({'alphas':ccpnew,'train scores':train_scores,'test score':test_scores})
best_alphas.sort_values(by= 'test score')
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccpnew, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccpnew, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()
# thuc hanh
data = pd.read_csv('C:/Users/admin/Desktop/smarket.csv',header = 'infer',index_col=None)
df = data.copy()
x= df.iloc[:,2:7]
y = df.iloc[:,9]
xtrain,xtest,ytrain,ytest =  sklean.model_selection.train_test_split(x,y,test_size=0.2)
Tree = tree.DecisionTreeClassifier(random_state=0)
model = Tree.fit(xtrain,ytrain)
path = model.cost_complexity_pruning_path(xtrain, ytrain)
ccp_alpha = path.ccp_alphas
ccp_alpha
models = []
for i in ccp_alpha :
    Tree = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=i)
    model = Tree.fit(xtrain,ytrain)
    models.append(model)
node = [model.tree_.node_count for model in models]
depth = [model.tree_.max_depth for model in models]

fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alpha, node, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alpha, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

#%% ket luan khoang 9 node
list1 = {'alpha': ccp_alpha,'node':node,'depth':depth}
prune = pd.DataFrame(list1)
prune
train_scores = [model.score(xtrain, ytrain) for model in models]
test_scores = [model.score(xtest, ytest) for model in models]

fig,ax = plt.subplots(figsize = (10,10))
ax.plot(ccp_alpha,train_scores,label = 'train_scores')
ax.plot(ccp_alpha,test_scores,label = 'test_scores')
ax.set_xlabel('ccp_alpha')
ax.set_ylabel('train_scores')
ax.set_title('train test')
ax.legend()
plt.show()
#%% 
ree = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.003403)
model_xin = ree.fit(xtrain,ytrain)
ypred = model_xin.predict(xtest)
sklean.metrics.confusion_matrix(ytest, ypred)
fig = plt.figure(figsize=(10,20))
_=tree.plot_tree(model_xin,feature_names=xtest.columns.values,
               class_names=ytest.values,filled=True)
ytrain.values
from dtreeviz.trees import dtreeviz # remember to load the package
hi=tree.export_graphviz(model_xin,feature_names=Xtrain.columns.values.tolist(),
               class_names=ytrain,filled=True)
import graphviz
graph = graphviz.Source(hi, format="png") 
graph

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
clf = tree.DecisionTreeClassifier(random_state=1234)
modelx = clf.fit(X, y)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)
iris.target_names
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,
                                filled=True)
iris.target_names
#%% random forest random forest ko phai tim cpp
data = pd.read_csv('C:/Users/admin/Desktop/smarket.csv',header = 'infer',index_col=None)
df = data.copy()
df
x = df.iloc[:,2:6]
y = df.iloc[:,9]
xtrain,xtest,ytrain,ytest = sklean.model_selection.train_test_split(x,y,test_size=0.2)
from sklearn import ensemble as rd
random_forest = rd.RandomForestClassifier(n_estimators=500,ccp_alpha=0.004903)
model = random_forest.fit(xtrain,ytrain)
ypred = model.predict(xtest)
model.decision_path(xtest)
sklean.metrics.confusion_matrix(ytest, ypred)
model.bootstrap
tree.plot_tree(model.estimators_[0],feature_names=xtest.columns.values.tolist(),class_names=ytest.values.tolist())
model.estimators_
