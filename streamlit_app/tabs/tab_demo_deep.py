from logging import PlaceHolder
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import label_binarize

import streamlit as st
import pandas as pd
import numpy as np
#import preprocessing as pp

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence, text
from keras.preprocessing.text import Tokenizer

from config import models_dir


title = "Demo - Model DL"
sidebar_name = "Demo - Model DL"

#model_path = "../models/prot_model_deep"

def load_models():
    #CNN Model
    cnn_model =  tf.keras.models.load_model('../models/model_deep_cnn1.pkl')
    
    #Tokenizer
    with open('../models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    #Label Binarizer
    with open('../models/labelbinarizer.pkl', 'rb') as handle:
        label_binarizer = pickle.load(handle)
    
    return cnn_model, tokenizer, label_binarizer

def dl_cnn_predict(seq, model, tokenizer, lb, maxlen=512):
    
    #tokenize sequence
    X = tokenizer.texts_to_sequences([seq])
    X = sequence.pad_sequences(X, maxlen=maxlen)

    #Make prediction
    y_pred = model.predict(X)
    #Inverse transform label
    y = lb.inverse_transform(y_pred)

    return y

def run():

    st.title(title)

    #open model
    model, tokenizer, lb = load_models()

    with st.container():
        st.subheader("Test DL Model")
        st.markdown("--------")

        # col1, col2 = st.columns(2)
        # with col1:
        #     with st.container():
        #         uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
        #     st.markdown("--------")
        #     with st.container():
        #         manual_input =  st.button('Saisie des paramètres')
        
        # with col2:
        # #st.markdown("--------")
        #     placeholder = st.empty()

        # if uploaded_file is not None:
        #     print(uploaded_file)
        #     dataframe = pd.read_csv(uploaded_file)
        #     with placeholder.container():
        #         with st.spinner("Please wait..."): 
        #             df = dl_predict_with_dataframe(dataframe)
                
        #         st.dataframe(data=df[['predicted_labels']]) 
        #st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        # if manual_input:
        #     with placeholder.container():
    with st.form(key='user_deep_inputs'):
        col1, col2 = st.columns(2)
        with col1:
            sequence  = st.text_input('Sequence')
            submit_button  =  st.form_submit_button(label='Predict')
                        
        with col2:  
            placeholder = st.empty()
            if submit_button:
                with placeholder.container():                     
                    #Make prediction
                    with st.spinner("Wait.."):
                        y_pred = dl_cnn_predict(sequence, model, tokenizer, lb) 
                        with placeholder.container():
                            st.write("Prediction class : ")
                            st.write(y_pred)
                            
                    


   






