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
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_validate
from sklearn import metrics
#!pip install --upgrade scikit-learn
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss,ConfusionMatrixDisplay,RocCurveDisplay


# UTILITY FUNCTIONS
def scoring_metrics(clf,y_test,y_pred,y_prob):
    """
    Given label ground truth and predictions 
    this function returns all scoring metrics 
    for a binary classification problem.
    """
    label_dict = {0:'benign',1:'malignant'}
    
    cm = confusion_matrix(y_test,y_pred,normalize='all')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {'confusion_matrix':cm,
            'Accuracy': accuracy_score(y_test,y_pred),
            'Precision': tp / (tp + fp),
            'Recall': tp / (tp + fn),
            'Jaccard': tp / (tp + fp + fn),
            'Dice': 2*tp / (2*tp + fp + fn),
            'Cross-entropy': log_loss([label_dict[x] for x in y_test],y_prob) if clf.startswith('Log') else np.nan}


# PREPROCESSING 
#------------------------------------------------------------------------
class Preprocessing:
    def __init__(self,dataset_name):
        self.df_name = dataset_name 
    
    def load_dataset(self,dataset_name):
        load_data = {'Diabetes Dataset':load_diabetes(as_frame=True),
                     'California Housing Dataset':fetch_california_housing(as_frame=True),
                     'Breast Cancer Wisconsin Dataset':load_breast_cancer(as_frame=True)}

        self.data = load_data[self.df_name]['data']
        self.target = load_data[self.df_name]['target'] 

        if self.df_name == 'Diabetes Dataset':
            renaming_dict={'age':'age','sex':'sex','bmi':'bmi','bp':'avg_bp','s1':'tc',
                            's2': 'ldl','s3': 'hdl','s4':'tch','s5':'ltg','s6':'glu'}
            self.data.rename(renaming_dict,inplace=True)

    
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

    def __init__(self): 
        self.b = 0

    def load_dataset(self,dataset_name):
        load_data = {'Diabetes Dataset':load_diabetes(as_frame=True),
                     'California Housing Dataset':fetch_california_housing(as_frame=True),
                     'Breast Cancer Wisconsin Dataset':load_breast_cancer(as_frame=True)}
        load_data_num = {'Diabetes Dataset':load_diabetes(),
                         'California Housing Dataset':fetch_california_housing(),
                         'Breast Cancer Wisconsin Dataset':load_breast_cancer()}

        self.data = load_data[dataset_name]['data']
        self.target = load_data[dataset_name]['target'] 
        self.X= load_data_num[dataset_name]['data']
        self.y = load_data_num[dataset_name]['target'] 

        if dataset_name == 'Diabetes Dataset':
            renaming_dict={'age':'age','sex':'sex','bmi':'bmi','bp':'avg_bp','s1':'tc',
                            's2': 'ldl','s3': 'hdl','s4':'tch','s5':'ltg','s6':'glu'}
            self.data.rename(renaming_dict,inplace=True)
    
    def generate_data(self,gen_options):
        
        self.generate_options = gen_options
        self.X, self.y = make_classification(**self.generate_options)
    
    
    def classify(self,options):
        
        self.scoring_metrics = ('accuracy',
                   'precision',
                   'recall',
                   'jaccard',
                   'roc_auc',
                   'log_loss')

        if options['clf_name']=='Logistic Regression':
            self.clf = LogisticRegression(C=options['C']) 
        elif options['clf_name']=='Support Vector Classifier':
            self.clf = SVC(C=options['C']) 

        if options['val_strategy']['name']=='Train - test split':
            
            # initialize figure
            fig_roc_base = plt.figure(figsize=(15,15))
            gs_roc_base = gridspec.GridSpec(1,1)
            ax_roc_base = {}

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,**options['val_strategy']['options'])
            
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            y_prob = self.clf.predict_proba(X_test)

            if len(np.unique(self.y)==2):
                # calculate metrics for binary classification
                results = scoring_metrics(options['clf_name'],y_test,y_pred,y_prob)
            
                # ROC curve
                ax_roc_base[0] = fig_roc_base.add_subplot(gs_roc_base[0,0])
                RocCurveDisplay.from_estimator(self.clf, X_test, y_test,ax=ax_roc_base[0])
                ax_roc_base[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.6)
                ax_roc_base[0].set_title(f'{options["clf_name"]}')
                ax_roc_base[0].legend(loc="lower right")
        

