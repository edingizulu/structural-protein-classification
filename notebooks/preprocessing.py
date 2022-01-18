
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from scipy.stats import skew


def load_datasets():
    # Open file 1
    df_prot = pd.read_csv('../data/data_no_dups.csv')
    # Open file 2
    df_seq = pd.read_csv('../data/data_seq.csv')
    # merge files 
    df = pd.merge(df_prot, df_seq, on =['structureId','macromoleculeType', 'residueCount' ],   how = 'inner')
    # return merged data
    return df



def prepare_target(df):
    """
    Removes NaN from Target
    Reduce modalities by filtering the values in the target 
    """
    if df.classification.isna().sum() > 0 : 
        #Remove NaN on target => put mode value
        df['classification'] = df.classification.fillna(df.classification.mode()[0])
        print(f'\033[1mComplete DataFrame has {df.shape[0]} lines et {df.shape[1]} columns \n Filter classification modalities')
    #Filter to a threshold of 5000 values
    df_filtered = PreprocessingTransformer.filter_classification(df, 5000)
    print(f'\033[1mFinal DataFrame has {df_filtered.shape[0]} lines et {df_filtered.shape[1]} columns after removing all classes with less than 5000 items')
    return df_filtered


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Protein Data Preprocessing class
    The class is built as transformer to be used in a pipeline
    
    Attributes
    ----------

    Methods
    -------
    

    """
            
    def fit(self, X, y=None):    
        #y.fillna(y.mode()[0], inplace=True)
        return self
    
    def transform(self, X) :
        print("1.Drop useless columns")
        to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails', 'publicationYear']
        X = X.drop([x for x in to_drop if x in X.columns], axis=1)
        print('2.Replace missing values in X')
        X = self.__handle_missing(X)
        print('3.Reduce modalities')
        X = self.__reduce_modalities(X)
        print("4.Correct skewness")
        X = self.__handle_skewness(X)
        print("5.scale and encode categ values")
        X = self.__scale_encode_data(X)
        print("-- Preprocessing done -- ")
        return X

    ##### Private methods 
    ######
    def __handle_missing(self, df):
        data = df.copy()
        for i in df.select_dtypes(exclude = np.number):
            data[i] = data[i].fillna(data[i].mode()[0])
        # for numerical values replace nan by median 
        data = data.fillna(df.select_dtypes(include = np.number).median())
        # drop anything else (useless)
        data = data.dropna()
        return data.reset_index(drop=True)

    def __reduce_modalities(self, df):
        df = df.copy()
        # analyse all variables separately
        df['experimentalTechnique'] = self.__map_experimentalTechique(df.experimentalTechnique)
        df['crystallizationMethod'] = self.__map_crystallizationMethod(df.crystallizationMethod)
        df['macromoleculeType']     = self.__map_macromoleculeType(df.macromoleculeType)
        df['phValue']               = self.__map_phValue(df.phValue)
        return df

    
    def __map_experimentalTechique(self, data):
        expt = (data.value_counts(normalize=True))
        techniques={}
        for tech in expt.index:
            if 'X-RAY DIFFRACTION' in tech:
                techniques[tech] = 'X-RAY DIFFRACTION' 
            else:
                techniques[tech] = 'others_tech_exp' 
        # return new experimentalTechniques values in dataframe
        return data.map(techniques)

    def __map_crystallizationMethod(self, data):
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
        
    def __map_macromoleculeType(self, data):
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

    @staticmethod
    def __ph_value(ph):
      if ph < 7.0:
          return 'acide'
      elif ph > 7.0:
          return 'basique'
      else:
          return 'neutre'

    def __map_phValue(self, data):
      #data.apply(PreprocessingTransformer.__ph_value, args=(self))
	    return data.map(lambda p: PreprocessingTransformer.__ph_value(p))


    def __handle_skewness(self, data) :
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

    
    def __scale_encode_data(self, data):
        df = data.copy()
        num_data = PreprocessingTransformer.__scaleData(df.select_dtypes(include = "number"))
        for col in num_data:
            df[col] = num_data[col]
        
        cat_data = df.select_dtypes(include = object)
        df = pd.get_dummies(df, prefix_sep= '_', drop_first=False, columns=cat_data.columns)
        return df

    @staticmethod
    def __scaleData(num_data):
        scaler = RobustScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(num_data), columns = num_data.columns)
        return data_scaled.reset_index(drop=True)


    @staticmethod
    def filter_classification(df, threshold=5000):
        """
        Removes all classes with count values < threshold

        Parameters
        ----------
        df : pd.DataFrame
            complete dataframe
        
        threshold : int
            minimum amount to keep

        """
        counts = df.classification.value_counts()
        types = np.asarray(counts[counts > threshold].index)
        return df[df.classification.isin(types)].copy()

    @staticmethod
    def map_classification(data, threshold):
        counts  = data.value_counts()
        up_thresh_indexes = np.asarray(counts[counts > threshold].index)
        classes={}
        for c in data:
            if c in up_thresh_indexes : 
                classes[c] = c
            else :
                classes[c]='other_classes'
        
        return data.map(classes)
