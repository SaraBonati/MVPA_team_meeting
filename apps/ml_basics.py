import numpy as np
import streamlit as st
import pandas as pd
import os
from .utilities import Classifier

# directory management
wdir = os.getcwd()
adir = os.path.join(wdir,'apps')
tdir = os.path.join(adir,'texts')

def app():
    st.title('ML 101')
    
    c1 = Classifier('first')
    
    col11, col12= st.columns(2)
    with col11:
        x1 = st.number_input('$x_{1}$', min_value=-5, max_value=5, value=1)
    with col12:
        y1 = st.number_input('$y_{1}$', min_value=-5, max_value=5, value=1)
    
    col21, col22= st.columns(2)
    with col21:
        x2 = st.number_input('$x_{2}$', min_value=-5, max_value=5, value=1)
    with col22:
        y2 = st.number_input('$y_{2}$', min_value=-5, max_value=5, value=1)

    #st.pyplot(c1.generate_data(x1,y1,x2,y2))
    st.plotly_chart(c1.generate_data(x1,y1,x2,y2))