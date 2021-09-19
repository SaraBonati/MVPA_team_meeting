# This script initializes MVPA journal club app using streamlit.
# The multipage-layout is taken from 
# Author: Sara Bonati (Plasticity group - FB-LIP @ MPI Berlin)
#----------------------------------------------------------------

# general utility import
import numpy as np
import pandas as pd
import os
import streamlit as st

# import specific apps
from apps import home
from multiapp import MultiApp

app = MultiApp()

# Add all apps here
app.add_app("Home", home.app)

app.run()
