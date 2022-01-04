import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import scipy as sc 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

#Tuning 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
#Metrics
from sklearn.model_selection import  learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import  f1_score,accuracy_score


from tqdm.notebook import tqdm, tqdm_notebook
import time


def handle_skewness(data: DataFrame) -> DataFrame :
    # calculate skewness for all numerical values in dataframe
    num_features = data.select_dtypes(include=np.number)
    for i in num_features:
        if abs(skew(num_features[i])) > 1 :
            #Replace data which has skewness > 1 by log(data)
            num_features[i]=num_features[i].map(lambda x: np.log(x) if x > 0 else 0)
    df = data
    if len(num_features.columns) != len(df.columns):
        #replace only num features in the provided dataset
        for i in num_features:
            df[i] = num_features[i]
    else:
        df =  num_features   
    return df

def scaleData(num_data:DataFrame) -> DataFrame :
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(num_data), columns = num_data.columns)
    return data_scaled


def evaluate_model_score(model, X, y):
    cv_result = []
    cv_means = []
    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=5)
    cv_result.append(cross_val_score(model, X, y = y, scoring = "accuracy", cv = kfold, n_jobs=4))
    cv_means.append(np.mean(cv_result))
    return cv_means


def train_models(models: list, X_train, y_train) -> dict :
    #X, y = df.loc[:, df.columns != 'classification'], df['classification']
    #X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state=123)

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

    ret_dict = {'train_acc_score' : train_acc_score, 'test_acc_score':test_acc_score,
                'train_f1_score': train_f1_score, 'test_f1_score': test_f1_score, 
                'train_pred': train_pred, 'test_pred' : test_pred}

    #return (train_acc_score, test_acc_score, train_f1_score, test_f1_score, train_pred, test_pred, test_pred_prob)
    return ret_dict


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize = (10,5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt.show()