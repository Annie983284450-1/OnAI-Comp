
#!/usr/bin/env python

# This file contains functions for evaluating algorithms for the 2019 PhysioNet/
# CinC Challenge. You can run it as follows:
#
#   python evaluate_sepsis_score.py labels predictions scores.psv
#
# where 'labels' is a directory containing files with labels, 'predictions' is a
# directory containing files with predictions, and 'scores.psv' (optional) is a
# collection of scores for the predictions.

################################################################################

# The evaluate_scores function computes a normalized utility score for a cohort
# of patients along with several traditional scoring metrics.
#
# Inputs:
#   'label_directory' is a directory of pipe-delimited text files containing a
#   binary vector of labels for whether a patient is not septic (0) or septic
#   (1) for each time interval.
#
#   'prediction_directory' is a directory of pipe-delimited text files, where
#   the first column of the file gives the predicted probability that the
#   patient is septic at each time, and the second column of the file is a
#   binarized version of this vector. Note that there must be a prediction for
#   every label.
#
# Outputs:
#   'auroc' is the area under the receiver operating characteristic curve
#   (AUROC).
#
#   'auprc' is the area under the precision recall curve (AUPRC).
#
#   'accuracy' is accuracy.
#
#   'f_measure' is F-measure.
#
#   'normalized_observed_utility' is a normalized utility-based measure that we
#   created for the Challenge. This score is normalized so that a perfect score
#   is 1 and no positive predictions is 0.
#
# Example:
#   Omitted due to length. See the below examples.

import numpy as np, os, os.path, sys, warnings
'''
this is for the online framework
each time it calculate the normalized utility score for the current patient only
'''

def evaluate_sepsis_score(label_directory, prediction_directory, patient_i):

    # Set parameters.
    '''
    SepsisLabel: indicates the onset of sepsis according to the Sepsis-3 definition, 
    where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number)
    indicate that there was no recorded measurement of a variable at the time interval.
    '''
    label_header       = 'SepsisLabel'
    prediction_header  = 'PredictedLabel'
    probability_header = 'PredictedProbability'
    # we reward classifiers that predict sepsis between 12 hours before and 3 hours after 
    # the onset time of sepsis
    dt_early   = -12
    # the optimal is that you can predict the sepsis 6 hours earlier
    dt_optimal = -6
    dt_late    = 3
            
   
    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.05
    u_tn     = 0

    # Find label and prediction files.
    # label_files = []
    all_test_files =  os.listdir(label_directory)
    # print(os.listdir(label_directory)[0])
    
    ## sort the files in the directory with the patient id 
    sorted_all_test_files = sorted(all_test_files)
    # print(sorted_all_test_files[0])
    # f_current_patient = all_test_files[patient_i]
    f_current_patient = sorted_all_test_files[patient_i]
    ## get the full path of the pxxxxxx.psv file
    label_file = os.path.join(label_directory, f_current_patient)
     
    print("processing label file:", label_file,"..........")
     
    prediction_file = os.path.join(prediction_directory, f_current_patient)
    
    

 
 
    # load the label & predictions & predicted_probability from the psv file
    labels       = load_column(label_file, label_header, '|')
    predictions  = load_column(prediction_file, prediction_header, '|')
    probabilities = load_column(prediction_file, probability_header, '|')
    
 
    num_rows = len(labels)
    
    # check if every element in this patient record is in the right scale
    for i in range(num_rows):
    
        if labels[i] not in (0,1):
            raise Exception('Label of ',f_current_patient, ' do not satify label ==0 or label ==1.')
        
        if predictions[i] not in (0,1):
            raise Exception('Prediction of ',  f_current_patient,' does not satisfy prediction == 0 or prediction == 1.')        
    
        if not 0 <= probabilities[i] <= 1:	
            warnings.warn('Probabilities of ',  f_current_patient,' do not satisfy 0 <= probability <= 1.'  )
    
    
    if 0< np.sum(predictions) < num_rows: # if this patient is predicted to develop sepsis ... 
        min_probability_positive = np.min(probabilities[predictions ==1])
        max_probability_negative = np.max(probabilities[predictions ==0])
    
    observed_predictions =  predictions    
    observed_utilities  = compute_prediction_utility(patient_i, labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    return observed_utilities

 

def compute_prediction_utility(patient_i,labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis? based on the ground truth
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels==1)  
        
        #print(labels)
        print("t_sepsis:",t_sepsis)
        print("patient id:",patient_i)
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels) # the total length of time the patient stayed in this hospital

    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP: true positive
            if is_septic and predictions[t]:
             
                if t_sepsis-12 <=t<=t_sepsis+3:
                    u = abs(1/15 * (t_sepsis - t))
                else:
                    u = 0
                     
            elif not is_septic and not predictions[t]:
                u =1

            elif not is_septic and predictions[t]:
                u =0
            elif is_septic and not predictions[t]:
            
                u =0        
 
    return u
     
 


def load_column(filename, header, delimiter):
    column = []
    with open(filename, 'r') as f:
    # read first row by row, then column be column
        for i, l in enumerate(f):
        
            arrs = l.strip().split(delimiter)
            if i == 0:
                try:
                    j = arrs.index(header)
                except:
                    raise Exception('{} must contain column with header {} containing numerical entries.'.format(filename, header))
            else:
                if len(arrs[j]):
                    column.append(float(arrs[j]))
    return np.array(column)

 

def compute_auc(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not 0 <= prediction <= 1:
                warnings.warn('Predictions do not satisfy 0 <= prediction <= 1.')

    # Find prediction thresholds.
    thresholds = np.unique(predictions)[::-1]
    if thresholds[0] != 1:
        thresholds = np.insert(thresholds, 0, 1)
    if thresholds[-1] == 0:
        thresholds = thresholds[:-1]

    n = len(labels)
    m = len(thresholds)

    # Populate contingency table across prediction thresholds.
    tp = np.zeros(m)
    fp = np.zeros(m)
    fn = np.zeros(m)
    tn = np.zeros(m)

    # Find indices that sort the predicted probabilities from largest to
    # smallest.
    idx = np.argsort(predictions)[::-1]

    i = 0
    for j in range(m):
        # Initialize contingency table for j-th prediction threshold.
        if j == 0:
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels)
            tn[j] = n - fn[j]
        else:
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

        # Update contingency table for i-th largest predicted probability.
        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Summarize contingency table.
    tpr = np.zeros(m)
    tnr = np.zeros(m)
    ppv = np.zeros(m)
    npv = np.zeros(m)

    for j in range(m):
        if tp[j] + fn[j]:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = 1
        if fp[j] + tn[j]:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = 1
        if tp[j] + fp[j]:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = 1
        if fn[j] + tn[j]:
            npv[j] = tn[j] / (fn[j] + tn[j])
        else:
            npv[j] = 1

    # Compute AUROC as the area under a piecewise linear function with TPR /
    # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
    # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
    # (y-axis).
    auroc = 0
    auprc = 0
    for j in range(m-1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc

 

def compute_accuracy_f_measure(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

    # Populate contingency table.
    n = len(labels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(n):
        if labels[i] and predictions[i]:
            tp += 1
        elif not labels[i] and predictions[i]:
            fp += 1
        elif labels[i] and not predictions[i]:
            fn += 1
        elif not labels[i] and not predictions[i]:
            tn += 1

    # Summarize contingency table.
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0

    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0

    return accuracy, f_measure

  
#if __name__ == '__main__':
    #auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(sys.argv[1], sys.argv[2])

    #output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(auroc, auprc, accuracy, f_measure, utility)
    #if len(sys.argv) > 3:
    #    with open(sys.argv[3], 'w') as f:
           # f.write(output_string)
   # else:
     #   print(output_string)
    
    #compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):


