import pandas as pd
import numpy as np, os, sys
import xgboost as xgb
import math
# The following two lines depend on the current path you are running your python
from Candidate_xgb.evaluate_sepsis_score import evaluate_sepsis_score
from Candidate_xgb.feature_engineering import feature_extraction
import csv

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w+') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

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
        test_pred[:, k] = y_test_pred
    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)

    return result_pro


# different from the offline algorithm, herein, each time we only need to predict one patient
def predict(psv, # original: data_set
            data_dir,
            save_prediction_dir,
            save_label_dir,
            model_path,
            risk_threshold
            ):

    '''
    for psv in data_set:
        patient = pd.read_csv(os.path.join(data_dir, psv), sep='|')
        features, labels = feature_extraction(patient)

        predict_pro = load_model_predict(features, k_fold = 5, path = './'+'Candidate_xgb/' + model_path + '/')
        PredictedProbability = np.array(predict_pro)
        PredictedLabel = [0 if i <= risk_threshold else 1 for i in predict_pro]

        save_prediction_name = save_prediction_dir + psv
        save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
        save_testlabel_name = save_label_dir + psv
        save_challenge_testlabel(save_testlabel_name, labels)
        |
        |
        |
        V
    '''

    ## modified (by @Anni Zhou) to the case in which there is only one patient
    patient = pd.read_csv(os.path.join(data_dir, psv), sep='|')
    features, labels = feature_extraction(patient)

    predict_pro = load_model_predict(features, k_fold = 5, path = './'+'Candidate_xgb/' + model_path + '/')

    # predict_pro = load_model_predict(features, k_fold = 5, path = './' + model_path + '/')
    PredictedProbability = np.array(predict_pro)
    PredictedLabel = [0 if i <= risk_threshold else 1 for i in predict_pro]

    save_prediction_name = save_prediction_dir + '/'+psv
    save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
    save_testlabel_name = save_label_dir +  '/'+psv
    save_challenge_testlabel(save_testlabel_name, labels)




'''
created by @Anni Zhou Mar 1st, 2021

get_reward_for_current_patient(test_set, test_data_path,prediction_directory, label_directory, patient_i )

	get the utility (defined as on that of the physionet official website)
	However, different from the offline scenario in the physionet 2019 sepsis challenge
	each time we only deal with one patient, i.e., each round in the UCB framework,
	so the round i (i.e., patient_i) should  be an input of this function

'''
def get_reward_for_current_patient(patient_i, experts_name_list):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    test_seti = test_set[patient_i]

    test_data_path = "./Candidate_xgb/data/all_dataset/"
    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( './prediction_xgb/',f_name)
        label_directory = os.path.join('./labels_xgb/',f_name)

    if not os.path.isdir(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)

    # prediction_directory = './Candidate_xgb/prediction_xgb/'
    # label_directory = './Candidate_xgb/label/'
   # model_path = 'Submit_model'
    model_path = 'xgb_model'
   # model_path = sys.argv[1]
   # predict(test_seti, test_data_path, prediction_directory, label_directory, model_path, 0.525)
    predict(test_seti, test_data_path, prediction_directory, label_directory, model_path, 0.525)

    # auroc, auprc, accuracy, f_measure,
    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)
    print("Calculating utility score.............\n")
    print("The utility of expert $1 (xgb--boost): ", patient_i)
    print(utility)
    print("\n")
    # return utility


    return float(utility)



if __name__ == "__main__":
    patient_i = 104

    test_set = np.load('./data/test_set.npy')

    print(type(test_set))
    print(test_set)
    print(np.sort(test_set,axis = None))
    test_set = np.sort(test_set,axis = None)

    N = len(test_set)



    test_data_path = "./data/all_dataset/"
    prediction_directory = './prediction_xgb/'
    label_directory = './label/'

    model_path = 'xgb_model'

    patient_i = 100
    test_seti = test_set[patient_i]
    print(test_seti)
    predict(test_seti, test_data_path, prediction_directory, label_directory, model_path, 0.525)
    utility = evaluate_sepsis_score(label_directory, prediction_directory, patient_i)

    print("The utility of expert1 (xgb--boost) -- no. of round: ", patient_i)
    print(utility)


'''
the original main funtion of the expert
we will not use it here,
because the original model is designed for offline framework
'''



'''
if __name__ == "__main__":
   # if len(sys.argv) != 2:
  #      raise Exception('Include the model directory as arguments, '
   #                     'e.g., python test.py Submit_model')

    test_set = np.load('./data/test_set.npy')
    test_data_path = "./data/all_dataset/"
    prediction_directory = './prediction/'
    label_directory = './label/'
    # model_path = sys.argv[1]
    model_path = 'Submit_model'

    predict(test_set, test_data_path, prediction_directory, label_directory, model_path, 0.525)

    auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(label_directory, prediction_directory)
    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(
                     auroc, auprc, accuracy, f_measure, utility)
    print(output_string)



    flag = os.path.isfile('../final_scores.csv')
    if flag:
        with open('final_scores.csv', mode='a+') as score_file:
            score_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            score_writer.writerow(['xgb' , auroc, auprc, accuracy, f_measure, utility])
    else:
        with open('final_scores.csv', mode='w+') as score_file:
            score_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            score_writer.writerow(['xgb' , auroc, auprc, accuracy, f_measure, utility])

 '''
