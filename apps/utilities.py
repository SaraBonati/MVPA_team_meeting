# In this script we code utility classes to support presentation slides
#----------------------------------------------------------------------
# Author @ Sara Bonati - Plasticity team meeting - 22/11/2021
#----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import os
import pandas as pd
from sklearn.datasets import load_diabetes,load_breast_cancer,fetch_california_housing

# PREPROCESSING 
#------------------------------------------------------------------------
class Preprocessing:
    def __init__(self,dataset_name):
        self.df_name = dataset_name 
    
    def load_dataset(self):
        load_data = {'Diabetes Dataset':load_diabetes(as_frame=True),
                     'California Housing Dataset':fetch_california_housing(as_frame=True),
                     'Breast Cancer Wisconsin Dataset':load_breast_cancer(as_frame=True)}

        self.data = load_data[self.df_name]['data']
        self.target = load_data[self.df_name]['target'] 

        if self.df_name == 'Diabetes Dataset':
            renaming_dict={'age':'age','sex':'sex','bmi':'bmi','bp':'avg_bp','s1':'tc',
                            's2': 'ldl','s3': 'hdl','s4':'tch','s5':'ltg','s6':'glu'}
            self.data.rename(renaming_dict,in_place=True)

        return self.data,self.target
    
    def plot_corr_matrix(self):
        # Compute the correlation matrix
        corr = self.data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Set up the matplotlib figure
        fig = plt.figure(figsize=(11, 9))
        gs = gridspec.GridSpec(1,1)
        ax={}
        ax[0] = fig.add_subplot(gs[0,0])
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=ax[0])
        
        return fig


# MODEL TRAINING - TESTING 
#------------------------------------------------------------------------
class Classifier:

    def __init__(self,name):
        self.name = name 
    
    def generate_data(self,x1,y1,x2,y2):
        X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)
    
    def preprocessing(self,pre_option):
        return None
    
    def classify(self,classifier):
        return None
