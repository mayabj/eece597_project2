###############################################################################################################################
# This script is used to find optimum Hyperparameters for neural network models
# We used a multilayer perceptron (MLP) architecture with 2 hidden layers of size n and n/2 neurons respectively
#
# 5-fold cross validation is performed for hyperparameter tuning
# We evaluated the MCC and the calibration metrics for all n ∈ {100, 200, ..., 800}
#
# The analysis is performed for 20 different decision threshold points (these are passed to the script from another script)
# Final hyperparameter values are decided based on the highest MCC (Test) values for all decision threshold points

##############################################################################################################

# Import packages

import argparse
import math
import numpy as np
import os
import re
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Declare Variables
header=["Decision Threshold", "hidden layer1", "fold", "True Positive (Train)",
        "True Negative(Train)","False Positive (Train)", "False Negative (Train)", "True Positive (Test)",
        "True Negative(Test)","False Positive (Test)", "False Negative (Test)", "Accuracy (Train)", "MCC (Train)", "ECE(Train)", "Brier_score (Train)", "NLL (train)", "Accuracy (Test)", "MCC (Test)", "ECE(Test)", "Brier_score (Test)", "NLL (Test)"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
k_fold_cross_validation = int(5) ## 5 fold cross validation
train_data_split = int(80) ## 80% of data in Train set
max_tree_depth_list = ['6']
no_of_trees_list = ['30']

# Specify File Paths
train_data_label_path_class = './project_data/circuit_split_train_data_for_cv/'
validate_data_label_path_class = './project_data/ttnkoi_class_50-1000_by_name/'

# ===================== Functions =================================================================

def capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class):
    """
    Capture circuit information from Binary Classification data
    @param train_data_label_path_class:
    @param train_data_file_class:
    @param train_label_file_class:
    @return : list_ckt_names_class
    """

    # regex_data = re.compile('.*(%s).*'%train_data_file_class)
    # regex_label = re.compile('.*(%s).*'%train_label_file_class)
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
    print('             No of Circuits in Classification data set : ', no_of_ckts_class)
    #print('Circuit Names :')
    #pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(list_ckt_names_class)
    return list_ckt_names_class


def organize_train_test_class(list_ckt_names_class,test_set,train_data_label_path_class):
    """
    Organize Binary Classification data
    @param list_ckt_names_class:
    @param test_set:
    @param train_data_label_path_class
    @return : data_train_X,data_train_y, data_test_X, data_test_y
    """
    train_set = list(set(list_ckt_names_class).difference(test_set))
    #print('         Test set : ', test_set)
    #print('         Train set : ')
    #pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(train_set)
    #print('created train set : ', time.strftime("%H:%M:%S", time.localtime()))
    test_file_root_list = np.char.add(train_data_label_path_class, test_set)
    test_data_file_list = np.char.add(test_file_root_list, train_data_file_class)
    test_label_file_list = np.char.add(test_file_root_list, train_label_file_class)

    train_file_root_list = np.char.add(train_data_label_path_class, train_set)
    train_data_file_list = np.char.add(train_file_root_list, train_data_file_class)
    train_label_file_list = np.char.add(train_file_root_list, train_label_file_class)

    columns_X = np.genfromtxt(train_data_file_list[0], delimiter=',', dtype=float).shape[1]

    # Concatenate info in csv files
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)
    data_test_X = np.empty((0, columns_X))
    data_test_y = np.empty(0)

    for file in train_data_file_list:
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in train_label_file_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    for file in test_data_file_list:
        data_test_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_test_X = np.concatenate((data_test_X, data_test_X_temp), axis=0)

    for file in test_label_file_list:
        data_test_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_test_y = np.concatenate((data_test_y, data_test_y_temp))

    return data_train_X, data_train_y, data_test_X, data_test_y

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
    #print('brier_score is :', brier_score)
    from sklearn.metrics import log_loss
    NLL = log_loss(labels, preds)
    return brier_score, NLL

def perform_k_fold_cross_validation(list_ckt_names_class, k_fold_cross_validation, test_ckts_array, gb):
    """
    Perform K-fold cross validation
    @param list_ckt_names_class:
    @param k_fold_cross_validation:
    @param test_ckts_array:
    @param gb:
    """
    tp_train_list = []
    tn_train_list = []
    fp_train_list = []
    fn_train_list = []
    tp_test_list = []
    tn_test_list = []
    fp_test_list = []
    fn_test_list = []
    mean_accuracy_train_list = []
    mean_accuracy_test_list = []
    routable_count_array_train_final = np.zeros(20)
    predictions_count_array_train_final = np.zeros(20)
    routable_count_array_test_final = np.zeros(20)
    predictions_count_array_test_final = np.zeros(20)


    for i in range(0, k_fold_cross_validation):
        print('           Fold : ',i)
        test_set = test_ckts_array[i]
        #print('test set:', test_set)
        data_train_X_class, data_train_y_class, data_test_X_class, data_test_y_class = organize_train_test_class(
            list_ckt_names_class, test_set, train_data_label_path_class)



        # Fit model to classification data
        clf = gb.fit(data_train_X_class, data_train_y_class)

        #print('  Inference/Predict')
        class_proba_train_array = clf.predict_proba(data_train_X_class)
        class_proba_test_array = clf.predict_proba(data_test_X_class)
        class_predicted_train = clf.predict(data_train_X_class).round()
        class_predicted_test = clf.predict(data_test_X_class).round()

        if i==0:
            ground_truth_array_train_final = data_train_y_class
            ground_truth_array_test_final = data_test_y_class
            class_proba_array_train_final = class_proba_train_array
            class_proba_array_test_final = class_proba_test_array
        else:
            ground_truth_array_train_final = np.concatenate((ground_truth_array_train_final,data_train_y_class))
            ground_truth_array_test_final = np.concatenate((ground_truth_array_test_final,data_test_y_class))
            class_proba_array_train_final = np.concatenate((class_proba_array_train_final,class_proba_train_array))
            class_proba_array_test_final = np.concatenate((class_proba_array_test_final,class_proba_test_array))

        mean_accuracy_train = accuracy_score(data_train_y_class, class_predicted_train)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(data_train_y_class, class_predicted_train).ravel()
        mcc_train = matthews_corrcoef(data_train_y_class, class_predicted_train)

        mean_accuracy_test = accuracy_score(data_test_y_class, class_predicted_test)
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(data_test_y_class, class_predicted_test).ravel()
        mcc_test = matthews_corrcoef(data_test_y_class, class_predicted_test)

        ece_train = expected_calibration_error_tempscale(data_train_y_class, class_proba_train_array, num_bins=20)
        ece_test = expected_calibration_error_tempscale(data_test_y_class, class_proba_test_array, num_bins=20)

        brier_score_train, nll_train = get_calibration_metrics(class_proba_train_array,data_train_y_class)
        brier_score_test, nll_test = get_calibration_metrics(class_proba_test_array, data_test_y_class)

        #print(header)
        print(str(decision_point) + ',' + str(hidden_layer1_size) + ',' + str(i) + ',' + str(tp_train) + ',' + str(tn_train) + ',' + str(fp_train) + ',' + str(fn_train) + ',' + str(tp_test) + ',' + str(tn_test) + ',' + str(fp_test) + ',' + str(fn_test) + ',' + str(mean_accuracy_train) + ',' + str(mcc_train) + ',' + str(ece_train) + ',' + str(brier_score_train) + ',' + str(nll_train) +',' + str(mean_accuracy_test) + ',' + str(mcc_test) + ',' + str(ece_test)+ ',' + str(brier_score_test) + ',' + str(nll_test) )

        tp_train = tp_train.astype(float)
        tn_train = tn_train.astype(float)
        fp_train = fp_train.astype(float)
        fn_train = fn_train.astype(float)
        tp_test = tp_test.astype(float)
        tn_test = tn_test.astype(float)
        fp_test = fp_test.astype(float)
        fn_test = fn_test.astype(float)

        tp_train_list.append(tp_train)
        tn_train_list.append(tn_train)
        fp_train_list.append(fp_train)
        fn_train_list.append(fn_train)
        tp_test_list.append(tp_test)
        tn_test_list.append(tn_test)
        fp_test_list.append(fp_test)
        fn_test_list.append(fn_test)

        mean_accuracy_train_list.append(mean_accuracy_train)
        mean_accuracy_test_list.append(mean_accuracy_test)

    tp_train_sum = sum(tp_train_list).astype(float)
    tn_train_sum = sum(tn_train_list).astype(float)
    fp_train_sum = sum(fp_train_list).astype(float)
    fn_train_sum = sum(fn_train_list).astype(float)
    tp_test_sum = sum(tp_test_list).astype(float)
    tn_test_sum = sum(tn_test_list).astype(float)
    fp_test_sum = sum(fp_test_list).astype(float)
    fn_test_sum = sum(fn_test_list).astype(float)

    mean_accuracy_train_final = np.mean(mean_accuracy_train_list)
    mean_accuracy_test_final = np.mean(mean_accuracy_test_list)

    mcc_train_final_term1 = (tp_train_sum * tn_train_sum) - (fp_train_sum * fn_train_sum)
    mcc_train_final_term2 = (tp_train_sum + fp_train_sum) * (tp_train_sum + fn_train_sum) * (tn_train_sum + fp_train_sum) * (tn_train_sum + fn_train_sum)
    mcc_test_final_term1 = (tp_test_sum * tn_test_sum) - (fp_test_sum * fn_test_sum)
    mcc_test_final_term2 = (tp_test_sum + fp_test_sum) * (tp_test_sum + fn_test_sum) * (tn_test_sum + fp_test_sum) * (tn_test_sum + fn_test_sum)
    mcc_train_final = mcc_train_final_term1/math.sqrt(mcc_train_final_term2)
    mcc_test_final = mcc_test_final_term1/math.sqrt(mcc_test_final_term2)

    ece_train_final = expected_calibration_error_tempscale(ground_truth_array_train_final, class_proba_array_train_final, num_bins=20)
    ece_test_final = expected_calibration_error_tempscale(ground_truth_array_test_final, class_proba_array_test_final, num_bins=20)
    brier_score_train_final, nll_train_final = get_calibration_metrics(class_proba_array_train_final, ground_truth_array_train_final)
    brier_score_test_final, nll_test_final = get_calibration_metrics(class_proba_array_test_final, ground_truth_array_test_final)

    print(str(decision_point) + ',' + str(hidden_layer1_size) + ',' + "cumulative" + ',' + str(
        tp_train_sum) + ',' + str(tn_train_sum) + ',' + str(fp_train_sum) + ',' + str(fn_train_sum) + ',' + str(
        tp_test_sum) + ',' + str(tn_test_sum) + ',' + str(fp_test_sum) + ',' + str(fn_test_sum) + ',' + str(
        mean_accuracy_train_final) + ',' + str(
        mcc_train_final) + ',' + str(ece_train_final) + ',' + ',' + str(brier_score_train_final) + ',' + str(nll_train_final) + str(mean_accuracy_test_final) + ',' + str(mcc_test_final) + ',' + str(
        ece_test_final)+ ',' + str(brier_score_test_final) + ',' + str(nll_test_final) )


# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++
now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--decision_point')
parser.add_argument('--hidden_layer1_size')
args = parser.parse_args()
d = args.decision_point
h = args.hidden_layer1_size
decision_point = int(d)
hidden_layer1_size = int(h)

#decision_point = 300
#hidden_layer1_size=100
hidden_layer2_size = int(hidden_layer1_size / 2)

print('\nDecision point : ', decision_point)
print('Max Depth values :', max_tree_depth_list)
print('No: of trees (n_estimators) : ', no_of_trees_list)
#print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
print('\nProgram Start Time : ',now.strftime("%Y-%m-%d %H:%M:%S"))
print('--------------------------------------------------------------------------------------')

print('\nCapture Binary Classification Data')
train_data_file_class = '-' + str(decision_point) + data_file_suffix
train_label_file_class = '-' + str(decision_point) + label_file_suffix
list_ckt_names_class = capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class)
list_ckt_names_class.sort()
print(list_ckt_names_class)

# K-fold cross validation
np.random.seed(0)
#np.random.shuffle(list_ckt_names_class)
# Creating 'k' test sets for k-fold cross validation after shuffling the list
print('\nCreate test sets for 5-fold cross validation')
test_ckts_array = np.array_split(list_ckt_names_class, k_fold_cross_validation)
print(test_ckts_array)

for m in max_tree_depth_list:
    max_tree_depth = int(m)
    for n in no_of_trees_list:
        no_of_trees = int(n)
        print('\nmax_depth : ', max_tree_depth)
        print('No: of trees (n_estimators) : ', no_of_trees)
        print('  Load Classification Model')
        print('    Hidden layer1 size : ', hidden_layer1_size)

        gb = MLPClassifier(
            hidden_layer_sizes=(hidden_layer1_size, hidden_layer2_size),
            random_state=0,
            max_iter=1000,
            tol=1e-6,
            n_iter_no_change=50
        )
        print('  Perform ' + str(k_fold_cross_validation) + '-fold cross validation and record metrics')
        print(header)
        perform_k_fold_cross_validation(list_ckt_names_class, k_fold_cross_validation, test_ckts_array, gb)

print('\n--------------------------------------------------------------------------------------')
#print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))
now = datetime.datetime.now()
print('\nProgram End Time : ',now.strftime("%Y-%m-%d %H:%M:%S"))

#ppdf.close()


