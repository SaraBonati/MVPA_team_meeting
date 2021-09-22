# In this script we code a simple Classifier class with methods dedicated
# to generating data, training a classifier from the ones offered in sklearn,
# and change classifier parameters as well as performing grid search for parameters.
# Author: Sara Bonati
# Plasticity team meeting - 19/11/2021

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd



class Classifier:

    def __init__(self,name):
        self.name = name 
    
    def generate_data(self):
        self.rv1 = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
        self.rv2 = multivariate_normal([1, 2], [[2.0, 0.3], [0.3, 0.5]])
        self.rv3 = multivariate_normal([-0.5, +0.5], [[2.0, 0.3], [0.3, 0.5]])

        self.samples1 = self.rv1.rvs(size=10)
        self.samples2 = self.rv2.rvs(size=10)
        self.samples3 = self.rv3.rvs(size=10)

        fig = plt.figure()
        gs = gridspec.GridSpec(1,1)
        ax = {}

        ax[0] = fig.add_subplot(gs[0,0])
        ax[0].scatter(self.samples1[:,0],self.samples1[:,1],s=3,c='b')
        ax[0].scatter(self.samples2[:,0],self.samples2[:,1],s=3,c='r')
        ax[0].scatter(self.samples3[:,0],self.samples3[:,1],s=3,c='g')

        return fig
    
    def preprocessing(self,pre_option):
        return None
    
    def classify(self,classifier):
        return None
