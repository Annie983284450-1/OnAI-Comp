import xgboost as xgb
import shap
import numpy as np, os, sys
import joblib
import pandas as pd
from treeinterpreter import treeinterpreter as ti
import lime
import lime.lime_tabular
if __name__ == "__main__":
    ## load model
    fitted_rf=  joblib.load("./saved_model/random_forest.sav")
    fitted_svc=  joblib.load("./saved_model/svc.sav")

    train = pd.read_csv('./data/train_rf.csv')
    test = pd.read_csv('./data/test_rf.csv')
# Dropping Features
    train.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
    "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
    test.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' ,'Lactate','Magnesium','Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
    "Glucose", 'Unit1', "Unit2", "HospAdmTime", 'Patient_id', 'DBP', 'SBP'], axis = 1, inplace = True)
# Convert categorical variables into dummy/indicator variables
    print(train.head())
    print("Convert categorical variables into dummy/indicator variables...")
    train_processed = pd.get_dummies(train)
    test_processed = pd.get_dummies(test)
    print(train_processed .head())
# Filling Null Values
    print("Filling Null Values...")
    train_processed = train_processed.fillna(train_processed.mean())
    test_processed = test_processed.fillna(test_processed.mean())
    print(train_processed.head())

# Create X_train,Y_train,X_test
    Y_train = train_processed.SepsisLabel
    Y_test = test_processed.SepsisLabel
    X_train = train_processed.drop(['SepsisLabel'],axis =1 )
    X_test = test_processed.drop(['SepsisLabel'],axis=1  )

    #
    # X_train.fillna(method='bfill' )
    # X_train.fillna(method = 'ffill' )
    # X_test.fillna(method='bfill' )
    # X_test.fillna(method = 'ffill' )



    #print(X_train.values)


    ## create an explainer
    # predict_fn_rf = lambda x: fitted_rf.predict_proba(x).astype(float)
    # X = X_train.values
    # explainer = lime.lime_tabular.LimeTabularExplainer(X,mode='classification',feature_names = X_train.columns,class_names=["sepsis","non-sepsis"])
    # test = X_test.insert(loc=0, column = 'SepsisLabel', value = Y_test)
    # print(test.head())

    #print(test.loc[[421]])
    # choosen_instance = X_test.loc[[421]].values[0]
    # exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
    # exp.show_in_notebook(show_all=False)

    #
    # prediction, bias, contributions = ti.predict(fitted_rf, X_test[0])

    #header = "features, label"
    #data = np.column_stack((features, labels))
    #np.savetxt( 'saved_modelfeatures, labels.dat' ,  data, fmt="%s", delimiter='|', header=header)
    #explainer_rf = shap.Explainer(fitted_rf)
    #shap_values_rf = explainer_rf(X_train)
    #shap.plots.waterfall(shap_values_rf[0])



    #xgb_model_path = sys.argv[1]
    #shap_data = shap_value(features, k_fold = 5, model_path = xgb_model_path)
    #shap.summary_plot(shap_data, features, max_display = 20, plot_type = "bar")
    #shap.summary_plot(shap_data, features, max_display = 20, plot_type = "dot")
