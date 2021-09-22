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

    st.pyplot(c1.generate_data())
