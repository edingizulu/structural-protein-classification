import numpy as np
import pandas as pd
import pickle

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
        if value < 6 :
            mTypes[m] = 'others_macro_mol'
        else :
            mTypes[m] = m

    return data.map(mTypes)

def _ph_value(ph):
    if (np.isnan(ph)):
        return  ph
    elif ph <= 7.0:
        return 'acide'
    elif ph > 7.0:
        return 'basique'
    else:
        return 'neutre'

def map_phValue(data):
    return data.map(_ph_value)


def clean_data(data : DataFrame) -> DataFrame:
    #handle missing
    df = handle_missing(data)

    # remove useless columns
    to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails']

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
    
    #print(df.head())
    return df

def handle_missing(df:DataFrame)->DataFrame:
    data = df.copy()
    #macromoleculeType and crystallizationMethod replace by mode
    data['macromoleculeType']     = data.macromoleculeType.fillna(data.macromoleculeType.mode()[0])
    data['crystallizationMethod'] = data.crystallizationMethod.fillna(data.crystallizationMethod.mode()[0])
    # for numerical values replace nan by median 
    data = data.fillna(df.select_dtypes(include = np.number).median())
    # drop other 
    data = data.dropna()    
    return data    
    

def save(filePath, data):
    with open(filePath, 'wb') as f : 
        my_pickle = pickle.Pickler(f)
        my_pickle.dump(data)


