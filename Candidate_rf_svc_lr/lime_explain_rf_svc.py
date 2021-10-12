import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
if __name__ == "__main__":
    fitted_rf=  joblib.load("./saved_model/random_forest.sav")
    fitted_svc=  joblib.load("./saved_model/svc.sav")
    X_train = pd.read_csv('./data/train_rf.csv')
    X_train.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
    "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
    X_train.fillna(method='bfill',inplace = True)
    X_train.fillna(method = 'ffill', inplace = True)
    Y_train = X_train.SepsisLabel
    X_train.drop(['SepsisLabel'],axis =1 , inplace = True )


    c = make_pipeline(X_train, fitted_rf)
    # idx = 83
    # exp = explainer.explain_instance(X_train[idx], c.predict_proba, num_features=6)
    # print('Patient id: %d' % idx)
    # print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
    # print('True class: %s' % class_names[newsgroups_test.target[idx]])
    #
    #
    # explainer = LimeTextExplainer(class_names=class_names)
