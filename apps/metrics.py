import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
import os
import re
from .utilities import Classifier
from sklearn.model_selection import train_test_split,KFold

# directory management
wdir = os.getcwd()
adir = os.path.join(wdir,'apps')
tdir = os.path.join(adir,'texts')

def app():

    # initalize classifier obj
    Clf = Classifier()

    # choose dataset
    st.markdown("<p style='text-align: center;'> Let's classify and evaluate the predictions! </p>", unsafe_allow_html=True)
    df_name = st.selectbox( 'Which dataset would you like to choose?', ('Breast Cancer Wisconsin Dataset',
                                                                        'I wanna make my own dataset!'))
    # load dataset or generate one
    if df_name=='I wanna make my own dataset!':
        with st.expander("Let's choose some generated dataset options then :) "):
            # initialize dictionary of generating data options
            gen_args = {}
            gen_args['n_classes'] = st.number_input('How many classes do you want?',min_value=2,max_value=5,value=2,step=1)
            gen_args['n_samples'] = st.number_input('How many data points?',min_value=100,max_value=1000,value=100,step=1)
            gen_args['n_features'] = st.number_input('How many features?',min_value=5,max_value=50,value=20,step=1)
            gen_args['n_informative'] = st.number_input('How many informative features?',min_value=5,max_value=gen_args['n_features'],value=5,step=1)
            gen_args['n_redundant'] = st.number_input('How many redundant features?',min_value=0,max_value=5,value=2,step=1)
            weights_text = st.text_input('The proportions of samples assigned to each class? (please separate numbers with space!)')
            if len(weights_text)>=(gen_args['n_classes']*2):
                gen_args['weights'] = [float(i)/gen_args['n_samples'] for i in re.findall(r"[-+]?\d*\.\d+|\d+", weights_text)]
            gen_args['class_sep'] = st.number_input('How easy do you want the classification to be (larger value -> easier)?',min_value=1.0,max_value=4.0,value=1.0,step=0.25)
            gen_args['flip_y'] = st.number_input('The fraction of samples whose class is assigned randomly?',min_value=0.0,max_value=0.5,value=0.01,step=0.01)
            
            st.button("Ok let's generate!",on_click=Clf.generate_data(gen_args))
                
    else:
        Clf.load_dataset(df_name)
    
    # choose validation arguments
    train_args = {}
    train_args['clf_name'] = st.selectbox('Which classifier would you like to use?',('Logistic Regression','Support Vector Classifier'))
    train_args['C'] = st.slider('Please choose a C value (the higher, the more we want to avoid misclassification)', min_value=0.05, max_value=1.0, value=0.5, step=0.05)
    train_args['val_strategy'] = {}
    val_strategy = st.radio('How do you want to validate your model?', ('Train - test split','KFold'))
    train_args['val_strategy']['name'] = val_strategy

    if val_strategy=='Train - test split':
        t = st.number_input('What proportion of data do you want to use for test?',min_value=0.2,max_value=0.5,value=0.2,step=0.05)
        train_args['val_strategy']['options'] = {'test_size':t,'shuffle':True,'random_state':42}
       
    elif val_strategy=='KFold':
        n_folds = st.number_input('How many folds?',min_value=3,max_value=10,value=3,step=1)
        train_args['val_strategy']['options'] = KFold(n_splits=n_folds,shuffle=True,random_state=42)

    # training and testing
    if st.button("Classify!"):
        figure,metrics = Clf.classify(train_args)
        if metrics is not None:
            # show figure
            st.pyplot(figure)
            # show metrics
            st.json(metrics)

    
        


