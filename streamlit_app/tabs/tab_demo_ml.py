import numpy as np
import pandas as pd
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import preprocessing as pp

from config import models_dir


title = "Demo - Model ML"
sidebar_name = "Demo - Model ML"

model_path = models_dir + "prot_clf.joblib"
model_reduced_path= models_dir + "prot_rfe_clf.joblib"

columns_clf = [ 'residueCount', 'resolution', 'structureMolecularWeight',
                'crystallizationTempK', 'densityMatthews', 'densityPercentSol',
                'macromoleculeType_DNA', 'macromoleculeType_Protein',
                'macromoleculeType_RNA', 'phValue_acide', 'phValue_basique',
                'phValue_neutre']

columns_rfe_clf = ['residueCount', 'resolution', 'structureMolecularWeight','crystallizationTempK', 'densityMatthews', 'densityPercentSol']



#Input   : Dataframe to test
#Returns : Dataframe with prediction column, accuracy score
def ml_predict_with_dataframe(df):
    #Open the model file
    model = joblib.load(model_path)
    print("model opened")
    #make the prediction
    data = df
    #do this before providing the Dataframe
    data = data.dropna().reset_index(drop=True)
    y = 0
    if 'classification' in data.columns : 
        y = data.classification
        data = df.drop('classification', axis=1)

    to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails', 'publicationYear', 'crystallizationMethod','experimentalTechnique']
    data = data.drop([x for x in to_drop if x in data.columns], axis=1)
    prep = pp.PreprocessingTransformer()
    data = prep.handle_missing(data)
    data = prep.reduce_modalities(data)
    data = prep.handle_skewness(data)
    data = prep.scale_encode_data(data)

    for col in columns_clf:
        if col not in data.columns:
            data[col] = 0

    data = data[columns_clf]

    print("Do the prediction: ")
    y_pred = model.predict(data)
    print("Prediction done")

    #insert predictions and give array back
    data['predicted_labels'] = y_pred

    return data

def ml_predict_with_user_input(input_dict):
    #Open the model with parameters reduced to 6
    model = joblib.load(model_reduced_path)

    #df = pd.DataFrame.from_dict(input_dict) #, orient='index', columns=['residueCount', 'resolution', 'structureMolecularWeight','crystallizationTempK', 'densityMatthews', 'densityPercentSol'])
    df = pd.DataFrame(input_dict, index=[0])
    y_pred = model.predict(df)

    return y_pred


def run():

    st.title(title)
    
    with st.container():
        st.markdown("--------")
        st.subheader("Charger un fichier de test")

        with st.container():
            uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
            placeholder = st.empty()

            if uploaded_file is not None:
                print(uploaded_file)
                dataframe = pd.read_csv(uploaded_file)
                with placeholder.container():
                    with st.spinner("Please wait..."): 
                        df = ml_predict_with_dataframe(dataframe)
                    
                    st.dataframe(data=df[['predicted_labels']]) 
            st.markdown("--------")

        with st.container():
            #manual_input =  st.button('Saisie des paramètres')
            #st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            #if manual_input:
                #with placeholder.container():
            st.subheader("Saisie manuelle des paramètres")

            with st.form(key='user_inputs'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    residueCount  = st.number_input('residuecount')
                    resolution    = st.number_input('resolution')
                    
                with col2:  
                    crystallizationTempK     = st.number_input('crystallizationTempK')  
                    structureMolecularWeight = st.number_input('structureMolecularWeight')
                with col3:    
                    densityMatthews   = st.number_input('densityMatthews')
                    densityPercentSol = st.number_input('densityPercentSol')
                
                submit_button  =  st.form_submit_button(label='Predict')
                            
                if submit_button:                       
                    #['residueCount', 'resolution', 'structureMolecularWeight','crystallizationTempK', 'densityMatthews', 'densityPercentSol']
                    input_dict={}
                    input_dict['residueCount']             = residueCount
                    input_dict['resolution']               = resolution
                    input_dict['structureMolecularWeight'] = structureMolecularWeight
                    input_dict['crystallizationTempK']     = crystallizationTempK
                    input_dict['densityMatthews']          = densityMatthews
                    input_dict['densityPercentSol']        = densityPercentSol

                    #with placeholder.container():
                    #st.write(structureMolecularWeight)
                    #Make prediction
                    placeholder2 = st.empty()
                    with st.spinner("Wait.."):
                        predicted_class = ml_predict_with_user_input(input_dict)
                        print(predicted_class)
                        
                        with placeholder2.container() :
                            st.write("Predicted class : "+ predicted_class)
                    

                    
                    
                    


   






