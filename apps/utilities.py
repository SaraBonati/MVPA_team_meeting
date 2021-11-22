# In this script we code utility classes to support presentation slides
#----------------------------------------------------------------------
# Author @ Sara Bonati - Plasticity team meeting - 22/11/2021
#----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import seaborn as sns
import plotly.express as px
import os
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes,load_breast_cancer,fetch_california_housing
from sklearn.datasets import make_classification
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_validate
from sklearn import metrics
from sklearn.feature_extraction import image
#!pip install --upgrade scikit-learn
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss,ConfusionMatrixDisplay,RocCurveDisplay

# directory management
wdir = os.getcwd()
apps_dir = os.path.join(wdir,"apps")

# UTILITY FUNCTIONS
#------------------------------------------------------------------------
def scoring_metrics(clf,y_test,y_pred,y_prob):
    """
    Given label ground truth and predictions 
    this function returns all scoring metrics 
    for a binary classification problem.
    """

    cm = confusion_matrix(y_test,y_pred,normalize='all')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return cm,{'Accuracy': accuracy_score(y_test,y_pred),
                'Precision': tp / (tp + fp),
                'Recall': tp / (tp + fn),
                'Jaccard': tp / (tp + fp + fn),
                'Dice': 2*tp / (2*tp + fp + fn),
                'Cross-entropy': log_loss(y_test,y_prob) if clf.startswith('Log') else np.nan}

#------------------------------------------------------------------------
#------------------------------------------------------------------------
class Classifier:

    def __init__(self): 
        self.b = 0
    
    #------------------------------------------------------------------------
    # PREPROCESSING 
    #------------------------------------------------------------------------
    def load_dataset(self,dataset_name):
        
        load_data = {'Breast Cancer Wisconsin Dataset':load_breast_cancer(as_frame=True)}
        load_data_num = {'Breast Cancer Wisconsin Dataset':load_breast_cancer()}
        
        self.data = load_data[dataset_name]['data']
        self.target = load_data[dataset_name]['target'] 
        self.all = pd.concat([self.data,self.target],axis=1)
        self.X= load_data_num[dataset_name]['data']
        self.y = load_data_num[dataset_name]['target'] 

    
    def generate_data(self,gen_options):
        self.generate_options = gen_options
        self.X, self.y = make_classification(**self.generate_options)
    
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

    def features_plot(self):

        col_name = st.selectbox( 'Which feature is the most helpful in distinguishing the classes?', list(self.all.columns))

        fig = plt.figure(figsize = (15,15))
        gs = gridspec.GridSpec(1,1)
        ax = {}
        
        ax[0] = fig.add_subplot(gs[0,0])
        sns.distplot(self.all[self.all['target']==0][col_name], color='g', label = 'Class 0',ax=ax[0])
        sns.distplot(self.all[self.all['target']==1][col_name], color='r', label = 'Class 1',ax=ax[0])
        ax[0].set_xlabel(col_name,fontsize=18)
        ax[0].legend(loc='best',fontsize=20)
        fig.tight_layout()
        return fig

    def pca_demo(self,mode):

        if mode == 'num':
            pca = PCA()
            pca.fit(self.data)

            fig=plt.figure(figsize=(10,18))
            gs = gridspec.GridSpec(2,1)
            ax = {}
 
            ax[0]=fig.add_subplot(gs[0,0])
            ax[0].bar(np.arange(0,pca.components_.shape[0],1),height=np.multiply(pca.explained_variance_ratio_,100),width=0.7,color='dodgerblue',alpha=0.5,align='center')
            #ax[0].set_xlim([-1,sizes_plot[0]])
            ax[0].set_xlabel('Principal component',fontsize=17)
            ax[0].set_ylabel('Explained variance (%)',fontsize=17)
            ax[0].set_title(f'PCs found: {pca.components_.shape[0]}',fontsize=20)

            ax[1]=fig.add_subplot(gs[1,0])
            pca2 = PCA(n_components=2)
            X_new = pca2.fit_transform(self.data)
            ax[1].scatter(X_new[:,0],X_new[:,1],s=35,color=['lightcoral' if t==0 else 'darkorchid' for t in self.target])
            ax[1].set_title('After PCA transform with 2 PCs',fontsize=20)
            ax[1].set_xlabel('$z_1$',fontsize=17)
            ax[1].set_xlabel('$z_2$',fontsize=17)
            fig.tight_layout()
            return fig

    
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    # MODEL TRAINING - TESTING 
    #------------------------------------------------------------------------
    def classify(self,options):
        
        # define metrics to compute
        self.scoring_metrics = ('accuracy','precision','recall',
                                'jaccard','roc_auc','log_loss')

        if options['clf_name']=='Logistic Regression':
            self.clf = LogisticRegression(solver='liblinear',C=options['C']) 
        elif options['clf_name']=='Support Vector Classifier':
            self.clf = SVC(C=options['C'],probability=True) 

        if options['val_strategy']['name']=='Train - test split':
           
            # initialize figure
            fig_roc_base = plt.figure(figsize=(10,25))
            gs_roc_base = gridspec.GridSpec(2,1)
            ax_roc_base = {}

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,**options['val_strategy']['options'])
            # train and predict
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            y_prob = self.clf.predict_proba(X_test)

            if len(np.unique(self.y)==2):
                # calculate metrics for binary classification
                cm,results = scoring_metrics(options['clf_name'],y_test,y_pred,y_prob)
                
                # ROC curve
                ax_roc_base[0] = fig_roc_base.add_subplot(gs_roc_base[0,0])
                RocCurveDisplay.from_estimator(self.clf, X_test, y_test,ax=ax_roc_base[0])
                ax_roc_base[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.6)
                ax_roc_base[0].set_title(f'{options["clf_name"]}')
                ax_roc_base[0].legend(loc="lower right")

                ax_roc_base[1] = fig_roc_base.add_subplot(gs_roc_base[1,0])
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred,ax=ax_roc_base[1])
                ax_roc_base[1].set_title('Confusion matrix')
                return fig_roc_base,results
        
        elif options['val_strategy']['name']=='KFold':
            
            # initialize figure
            fig_roc_kfold = plt.figure(figsize=(22,22))
            gs_roc_kfold = gridspec.GridSpec(1,1)
            ax_roc_kfold = {}
            ax_roc_kfold[0] = fig_roc_kfold.add_subplot(gs_roc_kfold[0,0])
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            results = {}

            for i, (train, test) in enumerate(options['val_strategy']['options'].split(self.X, self.y)):
                
                self.clf.fit(self.X[train], self.y[train])
                # fit and predict
                self.clf.fit(self.X[train], self.y[train])
                y_pred = self.clf.predict(self.X[test])
                y_prob = self.clf.predict_proba(self.X[test])
                # calculate metrics
                cm,results[f'fold{i+1}']= scoring_metrics(options['clf_name'] ,self.y[test],y_pred,y_prob)
                
                viz = RocCurveDisplay.from_estimator(
                    self.clf,
                    self.X[test],
                    self.y[test],
                    name="ROC fold {}".format(i),
                    alpha=0.3,
                    lw=2,
                    ax=ax_roc_kfold[0],
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax_roc_kfold[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax_roc_kfold[0].plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=4,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax_roc_kfold[0].fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            ax_roc_kfold[0].set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                title="KFold ROC curve"
            )
            ax_roc_kfold[0].legend(loc="lower right",fontsize=19)
            
            return fig_roc_kfold,results