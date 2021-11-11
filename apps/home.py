import numpy as np
import streamlit as st
import pandas as pd
import os


# directory management
wdir = os.getcwd()
adir = os.path.join(wdir,'apps')
tdir = os.path.join(adir,'texts')

def app():
    st.markdown("<h1 style='text-align: center; color: blue;'>Team meeting 22/11/2021</h1>", unsafe_allow_html=True)

    f = open(os.path.join(tdir,'home_text.md'), 'r')
    fileString = f.read()
    st.markdown(fileString,unsafe_allow_html=True)

    if st.button('PS: click here for some free balloons!'):
        st.balloons()