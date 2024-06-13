##########################################################################################################
# This script is used for capturing calibration scores for each iteration
##########################################################################################################

# Import packages

import time
import argparse
import joblib
import os
import re
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

# Declare Variables
header=["Decision Threshold", "Iteration", "True Positive", "True Negative","False Positive", "False Negative", "Accuracy", "MCC", "ECE", "Brier_score", "NLL"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'

# Specify File Paths
model_path_class = './project_data/ml_models_class_default_merge_mvtottnkoi/'
model_path_reg = './project_data/ml_models_reg_default_merge_mvtottnkoi/'
train_data_label_path_class = './project_data/merge_mvto_ttnkoi/'

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

def find_calibration_for_iteration_10_10(out_filename, concatenated_arr,model_path_class, model_path_reg,decision_point, header):
    """
    Capture calibration metrices for each iteration
    @param out_filename:
    @param concatenated_list:
    @param model_path_class:
    @param model_path_reg:
    @param decision_point:
    @param header:
    """
    with open(out_filename, 'w', newline='') as fileptr:
        writer = csv.writer(fileptr)
        writer.writerow(header)
        max_iter = int(np.max(concatenated_arr[:, 0]))

        if (decision_point % 100 == 0):
            threshold_limit = int(decision_point + 1)
        else:
            threshold_limit = int((100 * ((int(decision_point / 100)) + 1)) + 1)

        print('setting threshold limit = ', threshold_limit)

        for i in range(1,max_iter+1):
            #Capture information for each iteration
            #print("i=", i)
            arr_i = concatenated_arr[concatenated_arr[:, 0] == i]
            # Split the concatenated array back into X and y_label
            arr_i_X = arr_i[:, :-1]
            arr_i_y = arr_i[:, -1]
            labels = np.unique(arr_i_y)

            num_samples = arr_i_X.shape[0]
            #print('samples = ', num_samples)
            predicted_class_array = np.zeros(num_samples)
            class_proba_array = np.zeros((num_samples, 2))

            for j in range(0, num_samples):
                feature_vector_X = arr_i_X[j]
                # print(feature_vector_X)
                chosen_threshold = int(0)
                for threshold in range(100, threshold_limit, 100):
                    #print('Checking Decision Thresold : ', threshold)
                    # load classification model and predict
                    class_model_file_name = 'mvtottnkoi-misc0-' + str(threshold) + '-1_gradb_class_default'
                    model_file_path_class = str(np.char.add(model_path_class, class_model_file_name))
                    model_load_var_class = joblib.load(model_file_path_class)
                    y_class_predicted = model_load_var_class.predict(feature_vector_X.reshape(1, -1))
                    if (y_class_predicted == 1.0):
                        chosen_threshold = threshold
                        # print('predicted routable at threshold = ', threshold)
                        break

                if (chosen_threshold == 0):
                    predicted_class_array[j] = 0
                    class_proba_array[j] = [1.0, 0.0]

                else:
                    # load regression model for the chosen_threshold
                    reg_model_file_name = 'mvtottnkoi-misc0-' + str(chosen_threshold) + '-1_gradb_reg_default'
                    model_file_path_reg = str(np.char.add(model_path_reg, reg_model_file_name))
                    model_load_var_reg = joblib.load(model_file_path_reg)
                    y_remain_predicted = model_load_var_reg.predict(feature_vector_X.reshape(1, -1)).round()
                    feature_vector_X_column_0 = feature_vector_X[0]
                    y_absolute_prediction_with_regressor = y_remain_predicted + feature_vector_X_column_0

                    # Translate the regression prediction to classification

                    if (y_absolute_prediction_with_regressor > decision_point):
                        predicted_class_array[j] = 0
                        class_proba_array[j] = [1.0, 0.0]
                    else:
                        predicted_class_array[j] = 1
                        class_proba_array[j] = [0.0, 1.0]

            #Calculate Metrics
            if(labels.shape[0]==1):
                mean_accuracy=0
                mcc  = 0
                ece = 1
                brier_score=1
                nll = 1
                tp=tn=fp=fn=0
                data_row = [decision_point, i, tp,tn,fp,fn,mean_accuracy,mcc,ece,brier_score,nll]
                writer.writerow(data_row)
            else:
                mean_accuracy = accuracy_score(arr_i_y, predicted_class_array)
                tn, fp, fn, tp = confusion_matrix(arr_i_y, predicted_class_array).ravel()
                mcc = matthews_corrcoef(arr_i_y, predicted_class_array)
                ece = expected_calibration_error_tempscale(arr_i_y, class_proba_array, num_bins=20)
                brier_score, nll = get_calibration_metrics(class_proba_array, arr_i_y)

                data_row = [decision_point, i, tp,tn,fp,fn,mean_accuracy,mcc,ece,brier_score,nll]
                writer.writerow(data_row)

        fileptr.close()


# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument('--decision_point', dest='decision_point', type=str, help='Add decision_point')
args = parser.parse_args()
d = args.decision_point
decision_point = int(d)

result_file_train = 'results_iter_reg_merge_10_10_train_' + str(decision_point) + '.csv'
result_file_test = 'results_iter_reg_merge_10_10_test_' + str(decision_point) + '.csv'

print('Decision point : ', decision_point)

print('\nCapture Binary Classification Data')
train_data_file_class = '-' + str(decision_point) + data_file_suffix
train_label_file_class = '-' + str(decision_point) + label_file_suffix
list_ckt_names_class = capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class)
list_ckt_names_class.sort()
print(list_ckt_names_class)

print('   Create Train-Test sets : 80-20')
# train_index = round((len(list_ckt_names_class) * train_data_split)/100)
train_index = int(80)
train_ckt_list = list_ckt_names_class[:train_index]
test_ckt_list = list_ckt_names_class[train_index:]
print('      Circuits in Train set : ', len(train_ckt_list))
print('      Circuits in Test set : ', len(test_ckt_list))
#print(test_ckt_list)
data_train_X_class, data_train_y_class = organize_train_data(train_ckt_list, train_data_label_path_class,
                                                             train_data_file_class, train_label_file_class)
data_test_X_class, data_test_y_class = organize_train_data(test_ckt_list, train_data_label_path_class,
                                                           train_data_file_class, train_label_file_class)

#Concatenate features and labels
concatenated_train = np.concatenate((data_train_X_class, data_train_y_class[:, None]), axis=1)
concatenated_test = np.concatenate((data_test_X_class, data_test_y_class[:, None]), axis=1)

find_calibration_for_iteration_10_10(result_file_train,concatenated_train,model_path_class, model_path_reg,decision_point, header)
find_calibration_for_iteration_10_10(result_file_test,concatenated_test,model_path_class, model_path_reg,decision_point, header)

print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))



