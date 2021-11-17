import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
import os
from .utilities import Classifier, Preprocessing

# directory management
wdir = os.getcwd()
adir = os.path.join(wdir,'apps')
tdir = os.path.join(adir,'texts')

def app():
    
    # choose dataset
    st.markdown("<p style='text-align: center;'>Let's first select a dataset: \
                each of these datasets is more suited to show some common preprocessing issues/steps</p>", unsafe_allow_html=True)
    df_name = st.selectbox( 'Which dataset would you like to choose?', ('Diabetes Dataset',
                                                                        'California Housing Dataset',
                                                                        'Breast Cancer Wisconsin Dataset'))
    # initalize preprocessing obj
    P = Preprocessing(df_name)
    # info on data
    P.load_dataset()

    st.markdown("<p style='text-align: center;'>How does the dataset look like? \
                    What's the correlation between different features? </p>", unsafe_allow_html=True)
    st.dataframe(P.data.head(10))
    # plot of feature correlation matrix
    st.pyplot(P.plot_corr_matrix())





    
    