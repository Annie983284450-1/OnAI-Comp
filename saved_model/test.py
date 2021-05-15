import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from os import listdir
import csv
import shutil, os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib

if __name__ == "__main__":
    
    print("Loading original training and testing dataset (following the same split as Candidate 1)............\n")
            
    '''    
    train_npy_full_path = "../data/train_sepsis.npy"
    test_npy_full_path = "../data/test_set.npy"
    create_test_train_folder(train_npy_full_path, test_npy_full_path)
    preprocessing_rf(train_npy_full_path, test_npy_full_path)
    '''
    
    print("Reading preprocessed testing and training csv files............\n")
     
    X_test = pd.read_csv('../data/test_rf.csv')
    
    print("We have ", X_test.shape[0], " test entries and ",X_test.shape[1], "features in total.\n")
    # print(X_test)
  


    
    X_test.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
    "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True) 
    # Remove variables with more than 82% of na.
     
    print("Removing variables with more than 82% of na.\n")    
     
 
    X_test.fillna(method='bfill',inplace = True)
    X_test.fillna(method = 'ffill', inplace = True)
    
    print("Sampled data after removing NaN values ...........\n")    
    print(X_test)
    
 
    Y_test = X_test.SepsisLabel
 
    X_test.drop(['SepsisLabel'],axis=1 , inplace = True)
        
    
    
    saved_model1 = "random_forest.sav"
    rf =  joblib.load(saved_model1)
    
    saved_model2 = "svc.sav"
    svc =  joblib.load(saved_model2)
        
    svc_disp = plot_roc_curve(svc, X_test.head(1000) , Y_test.head(1000) )
    plt.show()
    ax = plt.gca()
    rf_disp = plot_roc_curve(rf, X_test.head(1000) , Y_test.head(1000) , ax=ax, alpha=0.8)
    plt.show() 
    
    
    curPath = os.path.abspath(os.path.dirname(__file__))
    print(curPath)
    
    
    
    
    
    
