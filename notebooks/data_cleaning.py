import numpy as np
import pandas as pd
import pickle

import seaborn as sns

from pandas.core.frame import DataFrame


def load_datasets():
    # Open file 1
    df_prot = pd.read_csv('../data/data_no_dups.csv')
    # Open file 2
    df_seq = pd.read_csv('../data/data_seq.csv')
    # merge files 
    df = pd.merge(df_prot, df_seq, on =['structureId','macromoleculeType', 'residueCount' ],   how = 'inner')
    # return merged data
    return df


def map_experimentalTechique(data):
    expt = (data.value_counts(normalize=True))
    techniques={}
    for tech in expt.index:
        if tech != 'X-RAY DIFFRACTION':
            techniques[tech] = "others_tech_exp"
        else:
            techniques[tech] = tech 
    # return new experimentalTechniques values in dataframe
    return data.map(techniques)

def map_crystallizationMethod(data):
    data = data.str.replace(',', ' ')
    data = data.str.replace('-', ' ')
    data = data.str.lower()

    cm = (data.value_counts(ascending = False, normalize=True) *100)
    methods={}
    for method, value in zip(cm.index, cm.values):
        if value < 1.90 :
            methods[method] = 'others_cryst_method'
        else :
            methods[method] = method
    # methods
    # Replace experimentalTechnique values in dataframe
    return data.map(methods)
    
def map_macromoleculeType(data):
    mt = (data.value_counts(normalize=True)*100)
    # Les 3 premiers types constituent 97% des modalités
    # Keep the 3 first and regroup the rest in OTHERS
    mTypes={}
    for m, value in zip(mt.index, mt.values):
        if value < 1 :
            mTypes[m] = 'others_macro_mol'
        else :
            mTypes[m] = m

    return data.map(mTypes)

def filter_classification(data):
    counts = data.classification.value_counts()
    types = np.asarray(counts[counts > 5000].index)
    return data[data.classification.isin(types)].copy()

def map_classification(data):
    counts  = data.value_counts()
    up_5K_indexes = np.asarray(counts[counts > 5000].index)
    classes={}
    for c in data:
        if c in up_5K_indexes : 
            classes[c] = c
        else :
            classes[c]='other_classes'
    
    return data.map(classes)

def _ph_value(ph):
    #Keep NaNs to remove them? 
    #if (np.isnan(ph)):
    #    return  ph
    #el
    if ph < 7.0:
        return 'acide'
    elif ph > 7.0:
        return 'basique'
    else:
        return 'neutre'

def map_phValue(data):
    return data.map(_ph_value)


def show_missing(df):
    missing_value = df.isnull().sum().sort_values(ascending=False)
    missing_value_percent = round(missing_value * 100 / df.shape[0], 1)
    missing_value_t = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_t.rename(columns={ 0:'Missing Values',
                                                                  1:'% Value'})
    cm = sns.light_palette('red', as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return

def handle_missing(df:DataFrame)->DataFrame:
    data = df.copy()
    #macromoleculeType and crystallizationMethod replaced by mode
    data['macromoleculeType']     = data.macromoleculeType.fillna(data.macromoleculeType.mode()[0])
    data['crystallizationMethod'] = data.crystallizationMethod.fillna(data.crystallizationMethod.mode()[0])
    # for numerical values replace nan by median 
    data = data.fillna(df.select_dtypes(include = np.number).median())
    # drop other 
    data = data.dropna()
    return data.reset_index()

def clean_data(data : DataFrame) -> DataFrame:
    #handle missing
    df = handle_missing(data)

    # remove useless columns
    to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails', 'publicationYear']

    if 'sequence' in df.columns:
        df = df.drop(to_drop, axis=1)
    
    # analyse all variables separately
    df['experimentalTechnique'] = map_experimentalTechique(df.experimentalTechnique)
    df['crystallizationMethod'] = map_crystallizationMethod(df.crystallizationMethod)
    df['macromoleculeType']     = map_macromoleculeType(df.macromoleculeType)
    df['phValue']               = map_phValue(df.phValue)

    df['classification'] = df['classification'].str.lower()
    df['classification'] = df['classification'].str.replace('(', '/', regex=False)
    df['classification'] = df['classification'].str.replace(',', '/', regex=False)
    df['classification'] = df['classification'].str.replace(', ', '/', regex=False)
    df['classification'] = df['classification'].str.replace('/ ', '/', regex=False)
    df['classification'] = df['classification'].str.replace(')', '', regex=False)

    #df = filter_classification(df)
    df['classification'] = map_classification(df.classification)
    return df

def save(filePath, data):
    with open(filePath, 'wb') as f : 
        my_pickle = pickle.Pickler(f)
        my_pickle.dump(data)


