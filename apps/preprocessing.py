import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
import os
from .utilities import Classifier

# directory management
wdir = os.getcwd()
adir = os.path.join(wdir,'apps')
tdir = os.path.join(adir,'texts')

def app():
    
    # choose dataset
    st.markdown("<p style='text-align: center;'>Let's first select a dataset: \
                we'll look at the Breast Cancer Wisconsin Dataset</p>", unsafe_allow_html=True)
    
    # initalize preprocessing classifier
    C = Classifier()
    # info on data
    C.load_dataset('Breast Cancer Wisconsin Dataset')

    st.markdown("<p style='text-align: center;'>How does the dataset look like? \
                    What's the correlation between different features? </p>", unsafe_allow_html=True)
    
    # plot of feature correlation matrix
    st.dataframe(C.data.head(10))
    st.pyplot(C.plot_corr_matrix())
    st.pyplot(C.features_plot())
        
    st.markdown("<p style='text-align: center;'> What if we apply PCA? </p>", unsafe_allow_html=True)
    st.pyplot(C.pca_demo('num'))

    #st.markdown("<p style='text-align: center;'> Here's a visual example of PCA </p>", unsafe_allow_html=True)
    #st.pyplot(P.pca_demo()[0])
    #st.pyplot(P.pca_demo()[1])




    
    