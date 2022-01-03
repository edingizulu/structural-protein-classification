import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import scipy as sc
from scipy.sparse.construct import rand 
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE, f_regression, SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer, OneHotEncoder


#Load cleaned file

def load_data(pickle_file_path : str) -> pd.DataFrame:
    with open(pickle_file_path, 'rb') as f:
        my_file = pickle.Unpickler(f)
        df = my_file.load()
        return df


def handle_skewness(data: DataFrame) -> DataFrame :
    # calculate skewness for all numerical values in dataframe
    num_features = data.select_dtypes(include=np.number)
    df = num_features.copy()
    for i in df:
        if abs(skew(df[i])) > 1 :
            #Replace data which has skewness > 1 by log(data)
            num_features[i]=num_features[i].map(lambda x: np.log(x) if x > 0 else 0)
    
    return num_features


def scaleData(num_data:DataFrame) -> DataFrame :
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(num_data), columns = num_data.columns)
    return data_scaled



def RFE_feat_selection(data, target, n_features):
    model = LogisticRegression()
    model.fit(data, target)

    selector = RFE(estimator=model, n_features_to_select=n_features)
    selector.fit(data, target)
    mask = selector.get_support()
    return np.array(data.columns)[mask]


def RandomForest_feat_selection(data, target) -> pd.Series: 
    model = RandomForestClassifier(random_state=1, max_depth=10)
    model.fit(data, target)
    #features = data.columns
    forest_importances = pd.Series(model.feature_importances_, index=data.columns)
    return(forest_importances)


