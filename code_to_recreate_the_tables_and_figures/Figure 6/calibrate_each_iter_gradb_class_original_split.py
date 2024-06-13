##########################################################################################################
# This script is used for capturing calibration scores for each iteration
##########################################################################################################

# Import packages

import time
import joblib
import os
import re
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

# Declare Variables
header=["Decision Threshold", "Iteration", "True Positive", "True Negative","False Positive", "False Negative", "Accuracy", "MCC", "ECE", "Brier_score", "NLLc"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
model_file_suffix='-1_gradb_class_6d30t'

# Specify File Paths
model_path_class = 'G:/ProjectData/ml_models_class_6_30_numpy_1_22/'
train_data_label_path_class = 'G:/ProjectData/circuit_split_train_data_for_cv/'
test_data_label_path_class = 'G:/ProjectData/ttnkoi_class_50-1000_by_name/'
decision_points_list = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650','700','750','800','850','900','950','1000']

def capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class):
    """
    Capture circuit information from Binary Classification data
    @param train_data_label_path_class:
    @param train_data_file_class:
    @param train_label_file_class:
    @return : list_ckt_names_class
    """

    regex_data = re.compile(".*({}).*".format(train_data_file_class))
    regex_label = re.compile(".*({}).*".format(train_label_file_class))
    list_data_files_class = []
    list_label_files_class = []
    list_ckt_names_class = []
    # Capture circuit information from binary classification data
    for root, dirs, files in os.walk(train_data_label_path_class):
        for file in files:
            if regex_data.match(file):
                list_data_files_class.append(file)
                list_ckt_names_class.append(file.split(train_data_file_class)[0])
            if regex_label.match(file):
                list_label_files_class.append(file)

    no_of_ckts_class = len(list_ckt_names_class)
    return list_ckt_names_class


def organize_train_data(list_ckt_names_class,train_data_label_path_class, train_data_file_class, train_label_file_class):
    """
    Organize Binary Classification data
    @param list_ckt_names_class:
    @param train_data_label_path_class
    @return : data_train_X,data_train_y
    """
    train_set = list_ckt_names_class
    train_file_root_list = np.char.add(train_data_label_path_class, train_set)
    train_data_file_list = np.char.add(train_file_root_list, train_data_file_class)
    train_label_file_list = np.char.add(train_file_root_list, train_label_file_class)

    # Concatenate info in csv files
    columns_X = np.genfromtxt(train_data_file_list[0], delimiter=',', dtype=float).shape[1]
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)

    for file in train_data_file_list:
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in train_label_file_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    data_train_y=data_train_y.astype(int)
    return data_train_X, data_train_y

def expected_calibration_error_tempscale(labels, preds, num_bins=20):
    '''
        calculates the expected calibration error given the number of bins
        Args:
            labels: Integer true labels
            preds: prediction vector with probabilty estimates for each class. Example: [0.3, 0.7]
            num_bins: number of bins to create
    '''
    # confidence - max of the the softmax scores
    confidences, predictions = np.max(preds, axis=1), np.argmax(preds, axis=1)
    accuracies = predictions == labels

    # calculates bin boundaries
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        # Calculated |confidence - accuracy| in each bin
        in_bin = np.array(list(map(lambda x: x > bin_lower, confidences))) * np.array(
            list(map(lambda x: x <= bin_upper, confidences)))
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def get_calibration_metrics(preds, labels):
    '''
        calculates the Brier score (BS) and the negative log-likelihood (NLL) of the given predictions
        Args:
            labels: Integer true labels
            preds: prediction vector with probabilty estimates for each class. Example: [0.3, 0.7]
        returns tuple of brier_score, NLL
    '''

    from sklearn.metrics import brier_score_loss
    brier_score = brier_score_loss(labels, preds[:, [1]])
    from sklearn.metrics import log_loss
    NLL = log_loss(labels, preds)
    return brier_score, NLL

def find_calibration_for_iteration(out_filename, concatenated_arr,model_class, header):
    """
    Capture calibration metrices for each iteration
    @param out_filename:
    @param concatenated_list
    """
    with open(out_filename, 'w', newline='') as fileptr:
        writer = csv.writer(fileptr)
        writer.writerow(header)
        max_iter = int(np.max(concatenated_arr[:, 0]))

        for i in range(1,max_iter+1):
            #print('Inference/Predict')
            #Capture information for each iteration
            #print(i)
            arr_i = concatenated_arr[concatenated_arr[:, 0] == i]
            # Split the concatenated array back into X and y_label
            arr_i_X = arr_i[:, :-1]
            arr_i_y = arr_i[:, -1]

            class_proba_array_i = model_class.predict_proba(arr_i_X)
            class_predicted_array_i = model_class.predict(arr_i_X).round()

            #Calculate Metrics
            #print("Calculate Metrics")
            labels = np.unique(arr_i_y)
            if(labels.shape[0]==1):
                mean_accuracy=0
                mcc  = 0
                ece = 1
                brier_score=1
                nll = 1
                data_row = [decision_point, i, tp,tn,fp,fn,mean_accuracy,mcc,ece,brier_score,nll]
                writer.writerow(data_row)
            else:
                mean_accuracy = accuracy_score(arr_i_y, class_predicted_array_i)
                tn, fp, fn, tp = confusion_matrix(arr_i_y, class_predicted_array_i).ravel()
                mcc = matthews_corrcoef(arr_i_y, class_predicted_array_i)
                ece = expected_calibration_error_tempscale(arr_i_y, class_proba_array_i, num_bins=20)
                brier_score, nll = get_calibration_metrics(class_proba_array_i, arr_i_y)

                data_row = [decision_point, i, tp,tn,fp,fn,mean_accuracy,mcc,ece,brier_score,nll]
                writer.writerow(data_row)

        fileptr.close()


# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

for d in decision_points_list:
    decision_point = int(d)
    result_file_train='results_separate_train_'+str(decision_point)+'.csv'
    result_file_test='results_separate_test_'+str(decision_point)+'.csv'

    print('\nDecision point : ', decision_point)
    model_file_name_class = 'mvto-' + str(decision_point) + model_file_suffix
    model_file_path_class = str(np.char.add(model_path_class, model_file_name_class))
    print('Model name : ', model_file_path_class)

    #print('Load Classification Model')
    model_load_var_class = joblib.load(model_file_path_class)

    #print('Load Train and Test Data')
    data_file_class = '-' + str(decision_point) + data_file_suffix
    label_file_class = '-' + str(decision_point) + label_file_suffix
    list_train_ckt_names_class = capture_ckt_info_class(train_data_label_path_class, data_file_class, label_file_class)
    list_train_ckt_names_class.sort()
    list_test_ckt_names_class = capture_ckt_info_class(test_data_label_path_class, data_file_class, label_file_class)
    list_test_ckt_names_class.sort()

    data_train_X_class, data_train_y_class = organize_train_data(list_train_ckt_names_class,train_data_label_path_class, data_file_class, label_file_class)
    data_test_X_class, data_test_y_class = organize_train_data(list_test_ckt_names_class, test_data_label_path_class, data_file_class, label_file_class)

    #Concatenate features and labels
    concatenated_train = np.concatenate((data_train_X_class, data_train_y_class[:, None]), axis=1)
    concatenated_test = np.concatenate((data_test_X_class, data_test_y_class[:, None]), axis=1)

    print('Train')
    find_calibration_for_iteration(result_file_train,concatenated_train,model_load_var_class, header)
    print('Test')
    find_calibration_for_iteration(result_file_test,concatenated_test,model_load_var_class, header)

    print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))


