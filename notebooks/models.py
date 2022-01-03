import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import scipy as sc 
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Classification 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from xgboost import  XGBClassifier

#Tuning 
from sklearn.model_selection import GridSearchCV, learning_curve, KFold, ParameterGrid, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer, OneHotEncoder

#Metrics

from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, roc_auc_score, precision_score,accuracy_score,roc_curve

from tqdm.notebook import tqdm, tqdm_notebook
from time import sleep
import time

from features_selection import load_data


classifiers = ['KNeighbors','RandomForest', 'AdaBoostClassifier','GradientBoostingClassifier','DecisionTreeClassifier',
'ExtraTreesClassifier','BaggingClassifier','Xgboost', 'SVC','LinearDiscriminantAnalysis', 'MultiLayerPerceptron', 'LogistiqueRegression']


df = load_data("../data/data_reduced.pkl")


def evaluate_model_score(model, X, y):
    cv_result = []
    cv_means = []
    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=5)
    cv_result.append(cross_val_score(model, X, y = y, scoring = "accuracy", cv = kfold, n_jobs=4))
    cv_means.append(np.mean(cv_result))
    return cv_means


def train_models(models: list, df:DataFrame) -> dict :
    X, y = df.loc[:, df.columns != 'classification'], df['classification']
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    #kfold = StratifiedKFold(n_splits=5, shuffle=True)
    # Fitting all the models 
    #models = [KNeighborsClassifier(),RandomForestClassifier()]
                #AdaBoostClassifier(),GradientBoostingClassifier(),DecisionTreeClassifier(),ExtraTreesClassifier(),
                #BaggingClassifier(),XGBClassifier(),SVC(), LinearDiscriminantAnalysis(),MLPClassifier(),LogisticRegression() ]

    model_scores = {}
    for i in tqdm(models):
        model_scores[i.__class__.__name__] = evaluate_model_score(i, X_train, y_train)
    end = time.time()
    return model_scores


def build_models(model, X_train,X_test, y_train, y_test) :
    #ext = model

    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

    #y_pred = ext.predict(X_test)

    test_pred_prob = model.predict_proba(X_test)[:,1]
    test_pred= model.predict(X_test)
    train_pred= model.predict(X_train)

    #print(prob)
    #print(pred_test)
    #print(classification_report(y_test, pred_test))

    train_acc_score = accuracy_score(y_train, train_pred,normalize=True)
    test_acc_score  = accuracy_score(y_test, test_pred, normalize=True)
    train_f1_score  = f1_score(y_train, train_pred, average='weighted')
    test_f1_score   = f1_score(y_test, test_pred, average='weighted')

    return (train_acc_score, test_acc_score, train_f1_score, test_f1_score, train_pred, test_pred, test_pred_prob)

