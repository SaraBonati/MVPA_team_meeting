import numpy as np
import streamlit as st
import pandas as pd
import os


# directory management
wdir = os.getcwd()
adir = os.path.join(wdir,'apps')
tdir = os.path.join(adir,'texts')

def app():
    st.title('Home')

    f = open(os.path.join(tdir,'home_text.md'), 'r')
    fileString = f.read()
    st.markdown(fileString,unsafe_allow_html=True)