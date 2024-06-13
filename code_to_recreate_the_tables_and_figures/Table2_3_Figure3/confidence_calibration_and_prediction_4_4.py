##########################################################################################################
# This script is used for capturing calibration and performance scores for expectation based system
##########################################################################################################

# Import packages

import time
import joblib
import os
import re
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

# Declare Variables
header=["Decision Threshold", "True Positive", "True Negative","False Positive", "False Negative", "Accuracy", "MCC", "ECE", "Brier_score", "NLL"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'

# Specify File Paths
model_path_class = './project_data/ml_models_class_default/'
model_path_reg = './project_data/ml_models_reg_default/'
#test_data_label_path_class = './project_data/ttnkoi_class_50-1000_by_name/'
#test_data_label_path_class = './project_data/ttnkoi_class_75-975_by_name/'
test_data_label_path_class = './project_data/circuit_split_train_data_for_cv/'

#decision_points_list = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650','700','750','800','850','900','950','1000']
#decision_points_list = ['75', '125', '175', '225', '275', '325', '375', '425', '475', '525', '575', '625', '675','725','775','825','875','925','975']


def capture_test_data_label_class(data_label_path_class, data_file_class, label_file_class):
    """
    Capture circuit information from Binary Classification data
    @param data_label_path_class:
    @param data_file_class:
    @param label_file_class:
    @return : data_train_X, data_train_y
    """

    regex_data = re.compile(".*({}).*".format(data_file_class))
    regex_label = re.compile(".*({}).*".format(label_file_class))
    list_data_files_class = []
    list_label_files_class = []
    # Capture circuit information from binary classification data
    for root, dirs, files in os.walk(data_label_path_class):
        for file in files:
            if regex_data.match(file):
                list_data_files_class.append(file)
            if regex_label.match(file):
                list_label_files_class.append(file)

    #no_of_ckts_class = len(list_data_files_class)
    #print('             No of Circuits in Classification data set : ', no_of_ckts_class)

    data_files_list = np.char.add(data_label_path_class, list_data_files_class)
    label_files_list = np.char.add(data_label_path_class, list_label_files_class)
    data_files_list.sort()
    label_files_list.sort()
    #print(data_files_list)
    #print(label_files_list)

    # Concatenate info in csv files
    columns_X = np.genfromtxt(data_files_list[0], delimiter=',', dtype=float).shape[1]
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)

    for file in data_files_list:
        #print(file)
        #print(type(file))
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in label_files_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    return data_train_X, data_train_y

def expected_calibration_error_tempscale(labels, preds, num_bins=20):
    '''
        calculates the expected calibration error given the number of bins
        Args:
            labels: Integer true labels
            preds: prediction vector with probabilty estimates for each class. Example: [0.3, 0.7]
            num_bins: number of bins to create
    '''
    # confidence - max of the softmax scores
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

def predict_y_class_by_reg_4_4(data_test_X_class, model_path_class, model_path_reg,d):
    """
    predict absolute number of iterations
    @param data_test_X_class:
    @param model_path_class:
    @param model_path_reg:
    @param decision_point:
    @return : predicted_class_array,class_proba_array
    """

    num_samples = data_test_X_class.shape[0]
    predicted_class_array = np.zeros(num_samples)
    class_proba_array = np.zeros((num_samples, 2))
    
    decision_point = int(d)

    if (decision_point <= 150):
        threshold_list = [150]
    elif (decision_point <= 250):
        threshold_list = [150, 250]
    elif (decision_point <= 400):
        threshold_list = [150, 250, 400]
    else:
        threshold_list = [150,250,400,1000]

    print('threshold list is : ', threshold_list)
    #print('setting threshold limit = ', threshold_limit)

    for i in range(0, num_samples):
        feature_vector_X = data_test_X_class[i]
        #print(feature_vector_X)
        chosen_threshold = int(0)
        for threshold in threshold_list:
            #print('Checking Decision Thresold : ', threshold)
            # load classification model and predict
            class_model_file_name = 'mvto-' + str(threshold) + '-1_gradb_class_default'
            model_file_path_class = str(np.char.add(model_path_class, class_model_file_name))
            model_load_var_class = joblib.load(model_file_path_class)
            y_class_predicted = model_load_var_class.predict(feature_vector_X.reshape(1, -1))
            #print('y_class_predicted = ', y_class_predicted)
            # print('class predicted :', y_class_predicted)
            if (y_class_predicted == 1.0):
                chosen_threshold = threshold
                #print('predicted routable at threshold = ', threshold)
                break

        if (chosen_threshold == 0):
            predicted_class_array[i] = 0
            class_proba_array[i] = [1.0, 0.0]

        else:
            # load regression model for the chosen_threshold
            reg_model_file_name = 'mvto-' + str(chosen_threshold) + '-1_gradb_reg_default'
            model_file_path_reg = str(np.char.add(model_path_reg, reg_model_file_name))
            model_load_var_reg = joblib.load(model_file_path_reg)
            y_remain_predicted = model_load_var_reg.predict(feature_vector_X.reshape(1, -1)).round()
            feature_vector_X_column_0 = feature_vector_X[0]
            y_absolute_prediction_with_regressor = y_remain_predicted + feature_vector_X_column_0

            # Translate the regression prediction to classification
            if (y_absolute_prediction_with_regressor > decision_point):
                predicted_class_array[i] = 0
                class_proba_array[i] = [1.0, 0.0]
            else:
                predicted_class_array[i] = 1
                class_proba_array[i] = [0.0, 1.0]


    return predicted_class_array,class_proba_array

# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))

parser = argparse.ArgumentParser()
parser.add_argument('--decision_point', dest='decision_point', type=str, help='Add decision_point')
args = parser.parse_args()
decision_point = args.decision_point
print(header)

result_file_char='results_reg_'+str(decision_point)+'.csv'
#print('Decision threshold : ', decision_point)

#print('Load Classification Data')
data_file_class = '-' + str(decision_point) + data_file_suffix
label_file_class = '-' + str(decision_point) + label_file_suffix
data_test_X_class, data_test_y_class = capture_test_data_label_class(test_data_label_path_class, data_file_class,
                                                                         label_file_class)
predicted_class_array, predicted_class_proba_array = predict_y_class_by_reg_4_4(data_test_X_class, model_path_class, model_path_reg,decision_point)

#Calculate metrics'
mean_accuracy = accuracy_score(data_test_y_class, predicted_class_array)
tn, fp, fn, tp = confusion_matrix(data_test_y_class, predicted_class_array).ravel()
mcc = matthews_corrcoef(data_test_y_class, predicted_class_array)

ece = expected_calibration_error_tempscale(data_test_y_class, predicted_class_proba_array, num_bins=20)
brier_score, nll = get_calibration_metrics(predicted_class_proba_array, data_test_y_class)

print(str(decision_point) + ',' + str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(mean_accuracy) + ',' + str(mcc) + ',' + str(ece) + ',' + str(brier_score) + ',' + str(nll))
#print(str(decision_point) + ',' + str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(mean_accuracy) + ',' + str(mcc) + ',' + str(ece))
print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))



