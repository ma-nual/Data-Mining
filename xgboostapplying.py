import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV,learning_curve
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_curve,roc_auc_score,auc,precision_recall_curve,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

dataset = pd.read_csv('adult.csv')
print('Dataset shape:', dataset.shape)
dataset.head()
print(dataset.dtypes)
dataset.isin(['?']).sum()
dataset = dataset[~(dataset.astype(str) == '?').any(1)]
sns.countplot(dataset['income'])
dataset = dataset.drop(['education','relationship','fnlwgt'],axis=1)
dataset.replace('<=50K',0,inplace=True)
dataset.replace('>50K',1,inplace=True)
dataset['income'].isin(['0']).sum()
dataset['income'].isin(['1']).sum()
dataset.reset_index(drop=True, inplace=True)
df1 = dataset.loc[dataset['income'].isin(['0'])]
df2 = dataset.loc[dataset['income'].isin(['1'])]
df1 = df1.sample(frac=0.4)
dataset = pd.concat([df1,df2])
dataset.reset_index(drop=True, inplace=True)
dataset['income'].isin(['0']).sum()
dataset['income'].isin(['1']).sum()

plt.xticks(rotation=45)
sns.countplot(dataset['workclass'],hue=dataset['income'])
plt.xticks(rotation=45)
sns.countplot(dataset['marital.status'],hue=dataset['income'])
sns.countplot(dataset['race'],hue=dataset['income'])
sns.countplot(dataset['sex'],hue=dataset['income'])
dataset.loc[dataset['native.country'] != 'United-States','native.country'] = 'non-usa'
sns.countplot(dataset['native.country'],hue=dataset['income'])
plt.xticks(rotation=90)
sns.countplot(dataset['occupation'],hue=dataset['income'])
sns.barplot(x='education.num',y='income',data=dataset)
sns.FacetGrid(dataset, col='income').map(sns.distplot, "age")

dataset.replace('Female',1,inplace=True)
dataset.replace('Male',0,inplace=True)
dataset.replace('United-States',1,inplace=True)
dataset.replace('non-usa',0,inplace=True)
dataset.loc[dataset['race'] != 'White','race'] = 'others'
dataset.replace('White',1,inplace=True)
dataset.replace('others',0,inplace=True)
dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-AF-spouse'], 'Married')
dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed','Married-spouse-absent'], 'Single')
dataset.replace('Married',1,inplace=True)
dataset.replace('Single',0,inplace=True)
dataset['workclass'] = dataset.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
dataset['occupation'] = dataset.occupation.map({'Exec-managerial':0,'Machine-op-inspct':1,'Prof-specialty':2,'Other-service':3,'Adm-clerical':4,'Transport-moving':5,'Sales':6,'Craft-repair':7,'Farming-fishing':8,'Tech-support':9,'Protective-serv':10,'Handlers-cleaners':11,'Armed-Forces':12,'Priv-house-serv':13})
ss = MinMaxScaler()
scale_features = ['age','workclass','education.num','occupation','capital.gain','capital.loss','hours.per.week']
dataset[scale_features] = ss.fit_transform(dataset[scale_features])
dataset.head()

x = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)

import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

model.fit(x,y)
xgb.plot_importance(model)
plt.show()
x_train = x_train.drop(['race','native.country'],axis=1)
x_test = x_test.drop(['race','native.country'],axis=1)

model = xgb.XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = xgb.XGBClassifier(learning_rate = 0.4)
max_depth = [3,4,5,6,7,8,9,10]
min_child_weight = [1,2,3,4,5,6]
param_grid = dict(max_depth = max_depth,min_child_weight = min_child_weight)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = xgb.XGBClassifier(learning_rate = 0.4,max_depth = 3,min_child_weight = 1)
gamma = [0,1,2,3,4,5]
param_grid = dict(gamma = gamma)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = xgb.XGBClassifier(learning_rate = 0.4,max_depth = 3,min_child_weight = 1,gamma = 0)
subsample = [0.6,0.7,0.8,0.9]
colsample_bytree = [0.6,0.7,0.8,0.9]
param_grid = dict(subsample = subsample,colsample_bytree = colsample_bytree)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = xgb.XGBClassifier(learning_rate = 0.4,max_depth = 3,min_child_weight = 1,gamma = 0,subsample = 0.9,colsample_bytree = 0.6)
n_estimators = [100,200,500]
param_grid = dict(n_estimators = n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = xgb.XGBClassifier(learning_rate = 0.4,max_depth = 3,min_child_weight = 1,gamma = 0,subsample = 0.9,colsample_bytree = 0.6,n_emistrators = 100)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

cm1 = confusion_matrix(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt='.20g')
target_names = ['0', '1']
print(classification_report(y_test, predictions, target_names=target_names))

fpr,tpr,threshold = roc_curve(y_test, predictions)
roc_auc = auc(fpr,tpr)
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()