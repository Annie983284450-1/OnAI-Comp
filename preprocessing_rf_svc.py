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
import sys
# path是当前脚本路径的上两级目录
path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir,os.path.pardir))
# 添加上两级目录(如果自定义模块在上两级目录下)
sys.path.append(path)
 
from Candidate1.evaluate_sepsis_score import evaluate_sepsis_score

'''
write all the patient data in the train_patient.csv file
append the patient_id tro identify each patient
add a time index for the data of each row (i.e., the data measured in an hour)
'''

def create_test_train_folder(train_npy_full_path, test_npy_full_path):
    test_set = np.sort(np.load(test_npy_full_path))
    train_set = np.sort(np.load(train_npy_full_path))
    all_traincsv = listdir('../data/all_dataset')
    os.mkdir('../data/train_set_rf')
    os.mkdir('../data/test_set_rf')
     
    for f in train_set:
        shutil.copy(os.path.join('../data/all_dataset', f), '../data/train_set_rf' )
    
    for f in test_set:
        shutil.copy(os.path.join('../data/all_dataset', f), '../data/test_set_rf' )   
           
     

'''
preprocessing dataset
'''
def preprocessing_rf(train_npy_full_path, test_npy_full_path):
    # load the training and testing dataset, which are the same with candidate 1
    
    
    test_set = np.sort(np.load(test_npy_full_path))
    train_set = np.sort(np.load(train_npy_full_path))
    all_traincsv = listdir('../data/all_dataset')
 
 

    # create and write the headers of the file
    
    with open('../data/test_rf.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        with open('../data/all_dataset/'+ all_traincsv[0],'r') as csvinput:
            reader = csv.reader(csvinput, delimiter='|')
            all = []
            row = next(reader)
            row.append('Patient_id')
            row.append('time')
            row.append('is_HR')
            row.append('is_O2Sat')
            row.append('is_Temp')
            row.append('is_MAP')
            row.append('is_Resp')
            row.append('is_Age')
            row.append('is_Gender')
            row.append('is_ICULOS')
            
            all.append(row)
            writer.writerows(all)
    with open('../data/train_rf.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        with open('../data/all_dataset/'+ all_traincsv[0],'r') as csvinput:
            reader = csv.reader(csvinput, delimiter='|')
            all = []
            row = next(reader)
            row.append('Patient_id')
            row.append('time')
            row.append('is_HR')
            row.append('is_O2Sat')
            row.append('is_Temp')
            row.append('is_MAP')
            row.append('is_Resp')
            row.append('is_Age')
            row.append('is_Gender')
            row.append('is_ICULOS')
            all.append(row)
            writer.writerows(all)  
            
    with open('../data/train_rf.csv', 'a') as csvoutput:
#  csv.reader() Return a reader object which will iterate over lines in the given csvfile.
        writer = csv.writer(csvoutput, lineterminator='\n')


        for ind, csv_name in enumerate(train_set):
            with open('../data/train_set_rf/'+ csv_name,'r') as csvinput:
                reader = csv.reader(csvinput, delimiter='|')
                all = []
 
     
                row = next(reader)
                 
                for i,row in enumerate(reader):
                    row.append(ind)
                    row.append(i)
                    all.append(row)
                writer.writerows(all)                 

    with open('../data/test_rf.csv', 'a') as csvoutput:
#  csv.reader() Return a reader object which will iterate over lines in the given csvfile.
        writer = csv.writer(csvoutput, lineterminator='\n')


        for ind, csv_name in enumerate(test_set):
            with open('../data/test_set_rf/'+ csv_name,'r') as csvinput:
                reader = csv.reader(csvinput, delimiter='|')
                all = []
 
     
                row = next(reader)
                 
                for i,row in enumerate(reader):
                    row.append(ind)
                    row.append(i)
                    all.append(row)
                writer.writerows(all)        


    '''
    with open('../data/test_set_rf.csv', 'a') as csvoutput:
 
        writer = csv.writer(csvoutput, lineterminator='\n')
        

        for ind, csv_name in enumerate(all_traincsv):
            if csv_name in test_set:
                with open('../data/all_dataset/'+ csv_name,'r') as csvinput:
                    reader = csv.reader(csvinput, delimiter='|')
                    all = []
                    if ind == 0:
                        row = next(reader)
                        # print(row)
                        row.append('Patient_id')
                        row.append('time')
                        all.append(row)
                         
                    else:                        
                        row = next(reader)
                        
                    for i,row in enumerate(reader):
                        row.append(ind)
                        row.append(i)
                        all.append(row)
                    writer.writerows(all)
    '''   
