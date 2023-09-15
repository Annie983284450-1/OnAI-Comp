import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from os import listdir
import csv
import shutil, os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import plot_roc_curve
#from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib
import sys
import lime
import lime.lime_tabular
import math
#  ../../
path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir,os.path.pardir))
# add ../../
sys.path.append(path)

from Candidate_xgb.evaluate_sepsis_score import evaluate_sepsis_score

'''
write all the patient data in the train_patient.csv file
append the patient_id tro identify each patient
add a time index for the data of each row (i.e., the data measured in an hour)
'''
 
## save the predicted labels
def save_challenge_predictions(file, scores, labels):
    with open(file, 'w+') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
## save the true labels
def save_challenge_testlabel(file, labels):
    with open(file, 'w+') as f:
        f.write('SepsisLabel\n')
        for l in labels:
            f.write('%d\n' % l)

def load_model_predict(X_test, k_fold, path):
    "ensemble the five XGBoost models by averaging their output probabilities"
    test_pred = np.zeros((X_test.shape[0], k_fold))
    X_test = xgb.DMatrix(X_test)
    for k in range(k_fold):
        model_path_name = path + 'model{}.mdl'.format(k+1)
        xgb_model = xgb.Booster(model_file = model_path_name)
        y_test_pred = xgb_model.predict(X_test)
        test_pred[:, k] = y_test_predT1
    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)

    return result_pro



def predict(psv,
            save_prediction_dir,
            save_label_dir,
            X_testi,
            Y_testi,
            risk_threshold,
            saved_model
            ):
    ## load trained model (if we run the script in the root folder)

    fitted_model =  joblib.load(saved_model)

    predicted_pro  = fitted_model.predict_proba(X_testi)
    # we only need to know the pro of developing sepsis
    PredictedProbability = np.array(predicted_pro[:,1])
    print("Predicted Probability:", PredictedProbability)
    PredictedLabel = [0 if i <= risk_threshold else 1 for i in predicted_pro[:,1]]


    labels = Y_testi

    ## tested in main() for p000016, the file is successfully saved, same as Candidate_xgb
    save_prediction_name = save_prediction_dir + '/'+ psv
    save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
    save_testlabel_name = save_label_dir +  '/'+ psv
    save_challenge_testlabel(save_testlabel_name, labels)

def get_reward_for_current_patient_rf(X_testi,Y_testi, patient_i, experts_name_list ):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based oT1n the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]
    # if we run from the root directory ...
    #prediction_directory = './Candidate_rf_svc_lr/prediction/'
    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( './prediction_rf/',f_name)
        label_directory = os.path.join('./labels_rf/',f_name)

# 1.mkdir( path [,mode] )
#       作用：创建一个目录，可以是相对或者绝对路径，mode的默认模式是0777。
#       如果目录有多级，则创建最后一级。如果最后一级目录的上级目录有不存在的，则会抛出一个OSError。
#
#  2.makedirs( path [,mode] )
#       作用： 创建递归的目录树，可以是相对或者绝对路径，mode的默认模式也是0777。
#       如果子目录创建失败或者已经存在，会抛出一个OSError的异常，Windows上Error 183即为目录已经存在的异常错误。如果path只有一级，与mkdir一样。
# 解决方式：先创建上级目录
#error 13 permission denied

    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    #prediction_directory = './Candidate_rf_svc_lr/prediction_rf/'
    #label_directory = './Candidate_rf_svc_lr/label/'
    saved_model= "./Candidate_rf_svc_lr/saved_model/random_forest.sav"
    print("random forest prediction: ")
    predict(test_seti, prediction_directory, label_directory, X_testi,Y_testi ,0.525, saved_model)


    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)

    print("Calculating utility score.............\n")
    print("The utility of Random Forest): ", patient_i)
    print(utility)
    print("\n")

    return float(utility)



def get_reward_for_current_patient_SVC(X_testi,Y_testi, patient_i, experts_name_list ):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]

    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( './prediction_svc/',f_name)
        label_directory = os.path.join('./labels_svc/',f_name)

    if not os.path.isdir(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)
    # # if we run from the root directory ...
    saved_model= "./Candidate_rf_svc_lr/saved_model/svc.sav"
    print("SVC prediction: ")
    predict(test_seti, prediction_directory, label_directory, X_testi,Y_testi ,0.525, saved_model)

    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)
    print("Calculating utility score.............\n")
    print("The utility of SVC: ", patient_i)
    print(utility)
    print("\n")
    return float(utility)
## build the LIME explainer for random forest classifier if we get an extremely high or low utility

def get_reward_for_current_patient_LR(X_testi,Y_testi, patient_i, experts_name_list):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]

    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( './prediction_lr/',f_name)
        label_directory = os.path.join('./labels_lr/',f_name)

    if not os.path.isdir(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)

    saved_model= "./Candidate_rf_svc_lr/saved_model/LR.sav"
 
    predict(test_seti, prediction_directory, label_directory, X_testi,Y_testi ,0.525, saved_model)

    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)
    print("Calculating utility score.............\n")
    print("The utility of LR: ", patient_i)
    print(utility)
    print("\n")
    return float(utility)

def lime_explainer(expert_name,  X_testi, patient_i):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]

    if not os.path.exists('Lime_explaination'):
        os.makedirs('Lime_explaination')

    if expert_name == 'RandomForest':
        saved_model =  "./Candidate_rf_svc_lr/saved_model/random_forest.sav"
    elif expert_name == "SVC":
        saved_model =  "./Candidate_rf_svc_lr/saved_model/svc.sav"
    elif expert_name == "LR":
        saved_model =  "./Candidate_rf_svc_lr/saved_model/LR.sav"
    fitted_model = joblib.load(saved_model)
    predict_fn = lambda x: fitted_model.predict_proba(x).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_testi.values, mode='classification',feature_names = X_testi.columns, class_names=['Non-Sepsis','Sepsis'] )
    choosen_instance = X_testi.values[0]
    exp = explainer.explain_instance(choosen_instance, predict_fn, num_features=np.shape(X_testi)[1])

    ## save the html
    file_path = './Lime_explaination/'+test_seti.strip('.psv') + '_'+ expert_name+'.html'
    exp.save_to_file(file_path, labels=None, predict_proba=True, show_predicted_value=True )

    # save as txt
    list_of_tuples = exp.as_list()
    txt_filename = './Lime_explaination/'+test_seti.strip('.psv') + '_'+expert_name+'.txt'
    f = open(txt_filename, 'w')
    for t in list_of_tuples:
        line = ','.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()

if __name__ == "__main__":

    #print("Loading original training and test X_test = X_test.loc[X_test['Patient_id'] == patient_i]ing dataset (following the same split as Candidate 1)............\n")
     ## train_sepsis.npy&test_set.npy files have been got by "build_dataset.py" from Candidate_xgb
     ## so they are in a sorted ascending order as in Candidate_xgb's preprocessing

    # train_npy_sepsis = "../data/train_sepsis.npy"
    # train_npy_nosepsis = "../data/train_nosepsis.npy"
    # test_npy = "../data/test_set.npy"
    #create_test_train_folder(test_npy, train_npy_sepsis,train_npy_nosepsis)
    #preprocessing_rf(test_npy, train_npy_sepsis,train_npy_nosepsis)

    #print("Reading preprocessed testing and training csv files............\n")
    X_train = pd.read_csv('../data/train_rf.csv')
    #print("We have ", X_train.shape[0], " train entries and ",X_train.shape[1], "features in total.\n")
    X_test = pd.read_csv('../data/test_rf.csv')
    #print("We have ", X_test.shape[0], " test entries and ",X_test.shape[1], "features in total.\n")
    #
    # print(isinstance(X_train, pd.DataFrame))
    # print(isinstance(X_test, pd.DataFrame))
    # 34285 training samples
    T= len(X_train['Patient_id'].unique())
    ## 6051 testing samples
    T1 = len(X_test['Patient_id'].unique())
    Y_train = X_train.SepsisLabel
    X_train.drop(['SepsisLabel'],  axis =1,inplace = True)
    for t in np.arange(math.floor(T/100)):
        patient_i = t
        print("Round ****************",t,"******************")
        X_traini = X_test.loc[X_test['Patient_id'] == patient_i]
        print(isinstance(X_traini, pd.DataFrame))
        X_traini.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
		    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
		    'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
		    "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
            # print("Removing variables with more than 82% of na for Expert 2 \(random forest & SVC\) ... \n")
        X_traini.fillna(method = 'bfill', inplace = True)
        X_traini.fillna(method = 'ffill', inplace = True)

    # print(X_train.isnull().values.any())
    # for t in np.arange(math.floor(T1/100)):
    #     patient_i = t
    #     print("Round ****************",t,"******************")
    #     X_testi = X_test.loc[X_test['Patient_id'] == patient_i]
    #     X_testi.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
	# 	    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
	# 	    'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
	# 	    "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
    #         # print("Removing variables with more than 82% of na for Expert 2 \(random forest & SVC\) ... \n")
    #     X_testi.fillna(method = 'bfill', inplace = True)
    #     X_testi.fillna(method = 'ffill', inplace = True)
    # Y_test = X_test.SepsisLabel
    # X_test.drop(['SepsisLabel'], axis =1,  inplace = True)
    #         #for i in np.arange(self.k):
    # print(X_test.isnull().values.any())



    #
    # patient_i =46
    #
    # X_test = X_test.loc[X_test['Patient_id'] == patient_i]
    #
    # #X_testi = X_test.loc[X_test['Patient_id'] == patient_i]
    # #print(X_testi)
    # X_train.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    # 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
    # 'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
    # "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
    #
    #
    # X_test.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    # 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
    # 'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
    # "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
    # # Remove variables with more than 82% of na.
    #
    # print("Removing variables with more than 82% of na.\n")
    #
    # X_train.fillna(method='bfill',inplace = True)
    # X_train.fillna(method = 'ffill', inplace = True)
    # X_test.fillna(method='bfill',inplace = True)
    # X_test.fillna(method = 'ffill', inplace = True)
    #
    #
    #
    # Y_train = X_train.SepsisLabel
    # Y_test = X_test.SepsisLabel
    # X_train.drop(['SepsisLabel'],axis =1 , inplace = True )
    # X_test.drop(['SepsisLabel'],axis=1 , inplace = True)
    # print("Sampled data after removing NaN values ...........\n")
    # print(X_test)
    #
    #
    #
    #
    # ## load trained model
    # saved_model1 = "../saved_model/random_forest.sav"
    # rf =  joblib.load(saved_model1)
    #
    # saved_model2 = "../saved_model/svc.sav"
    # svc =  joblib.load(saved_model2)
    #
    # ## the predicted prob is 0.xx, how could I improve it to 0.xxxxxx?????????
    # predicted_pro  =  svc.predict_proba(X_test)
    # predicted_pro  =  rf.predict_proba(X_test)
    # num_entries  = np.shape(predicted_pro)[0]
    # risk_threshold = 0.525 # same as Candidate_xgb
    # PredictedLabel = [0 if i <= risk_threshold else 1 for i in predicted_pro[:,1]]
    #
    # # print(predicted_pro )
    # print(predicted_pro )
    # print(np.shape(predicted_pro))
    # print(PredictedLabel)
    # print(np.shape(PredictedLabel))
    # PredictedProbability = np.array(predicted_pro[:,1])
    # print(PredictedProbability)
    # print(np.shape(PredictedProbability))
    #
    # test_set = np.load('../data/test_set.npy')
    # ## sort all the psv files based on the number included in the psv filename (in ascending order)
    # test_set = np.sort(test_set,axis = None)
    # # Just want to get the psv file name herein
    # psv = test_set[patient_i]
    # save_prediction_dir = "../prediction/"
    # save_label_dir = "../label/"
    # save_prediction_name = save_prediction_dir + psv
    # save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
    # save_testlabel_name = save_label_dir + psv
    #
    # labels =  Y_test
    # save_challenge_testlabel(save_testlabel_name, labels)




    ### run the following code to train and save the trained model (offline modules)
    #train the random forest classifier

    # n_estimatorsint, default=100
    # The number of trees in the forest.

    # random_stateint, RandomState instance or None, default=None
    # Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True)
    # and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).

    # rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    # rf.fit(X_train, Y_train)
    # joblib.dump(rf,'../saved_model/random_forest.sav')


    # The Receiver Operator Characteristic (ROC) curve is an evaluation metric for binary classification problems.
    # It is a probability curve that plots the TPR against FPR at various threshold values and essentially separates the ‘signal’ from the ‘noise’.
    # The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.
    '''
    svc = SVC( probability=True,random_state=42)
    svc.fit(X_train, Y_train)
    joblib.dump(svc,'../saved_model/svc.sav')


    svc_disp = plot_roc_curve(svc, X_test, Y_test)
    plt.show()
    # Get the current Axes instance on the current figure matching the given keyword args, or create one.
    ax = plt.gca()
    rf_disp = plot_roc_curve(rf, X_test, Y_test, ax=ax, alpha=0.8)
    svc_disp.plot(ax=ax, alpha=0.8)
    plt.show()
    '''
