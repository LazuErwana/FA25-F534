# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:56:05 2025

@author: taadair
"""


# import packages
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Assume current working directory is /home/user/project
# Change to a subdirectory
os.chdir("Logistic Regression")

# Reading from Excel Files 
df = pd.read_excel('lending_clubFull_Data_Set.xlsx')

# Change back to FA25-F534 directory
os.chdir("..")

def stratified_sample(df, col_name, frac):
    # Group by the specified column and apply the sample function to each group
    # setting random_state for reproducibility
    return df.groupby(col_name).apply(lambda x: x.sample(frac=frac, random_state=42)).reset_index(drop=True)

# Example usage:
# Assuming 'df' has a column 'product_type'
sampled_df = stratified_sample(df, 'product_type', 0.2) # Sample 20% from each product type