###############################################################################################################################
# This script is used to find optimum Hyperparameters for GradientBoosting Classifier models
# Hyperparameters considered are :
#    (1) max depth of individual trees (max_depth)
#    (2) number of trees (n_estimators) in the ensemble
# max_depth is varied between 6 & 7_gradb_calibrate_for_each_iter_separate [as obtained from optimum depth analysis of small decision tree]
# no:of trees (n_estimators) is varied in a grid search approach : eg:3,5,10..100 (default n_estimators=100 for GradientBoost)

# 5-fold cross validation is performed for hyperparameter tuning and following metrics are calculated for both Train & Test data
# ==> Accuracy, True Positive(TP), True Negative(TN), False Positive(FP), False Negative(FN), MCC (Mathews Correlation Coefficient)
# For every hyperparameter combination (max_depth, no: of trees) MCC value is calculated using
# the cumulative values of TP, TN, FP, and FN from all folds.

# The analysis is performed for 20 different decision threshold points (these are passed to the script from another script)
# Final hyperparameter values are decided based on the highest MCC (Test) values for all decision threshold points
############################################################################################################################
# Import packages

import argparse
import math
import time
import csv
import numpy as np
import os
import re
import pprint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

# Declare Variables
header=["Decision Point", "Max depth", "No of Trees", "Fold", "Accuracy (Train)", "Accuracy (Test)",
        "True Positive (Train)", "True Negative (Train)", "False Positive (Train)", "False Negative (Train)",
        "True Positive (Test)", "True Negative (Test)", "False Positive (Test)", "False Negative (Test)",
        "MCC (Train)", "MCC (Test)"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
k_fold_cross_validation = int(5) ## 5 fold cross validation
max_tree_depth_list = ['6', '7']
#no_of_trees_list = ['10','30','50','70','90']
no_of_trees_list = ['5','20','30']

# Specify File Paths
train_data_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_X.csv'
train_label_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_y.csv'
train_data_label_path_class = 'G:/ProjectData/circuit_split_train_data_for_cv/'


# ===================== Functions =================================================================

def load_train_data_reg(train_data_file_path,train_label_file_path):
    """
    Reads in Train Data
    @param train_data_file_path:
    @param train_label_file_path:
    @return : data_train_X,data_train_y
    """
    data_train_X = np.genfromtxt(train_data_file_path, delimiter=',', dtype=float)
    data_train_y = np.genfromtxt(train_label_file_path, delimiter=',', dtype=int, usecols=0)

    # Find how many routing runs are captured in the Y_labels.csv
    # Number of routing runs = number of 1s
    number_of_runs_train = list(data_train_y.flatten()).count(1)
    print("             Number of routing runs in train data : ", number_of_runs_train)
    return data_train_X,data_train_y


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
    pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(train_set)
    #print('created train set : ', time.strftime("%H:%M:%S", time.localtime()))
    test_file_root_list = np.char.add(train_data_label_path_class, test_set)
    test_data_file_list = np.char.add(test_file_root_list, train_data_file_class)
    test_label_file_list = np.char.add(test_file_root_list, train_label_file_class)

    train_file_root_list = np.char.add(train_data_label_path_class, train_set)
    train_data_file_list = np.char.add(train_file_root_list, train_data_file_class)
    train_label_file_list = np.char.add(train_file_root_list, train_label_file_class)

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


def perform_k_fold_cross_validation(list_ckt_names_class, k_fold_cross_validation, test_ckts_array, writer, gb):
    """
    Perform K-fold cross validation
    @param list_ckt_names_class:
    @param k_fold_cross_validation:
    @param test_ckts_array:
    @param writer:
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
    for i in range(0, k_fold_cross_validation):
        print('           Fold : ',i)
        test_set = test_ckts_array[i]
        # print('test set:', test_set)
        data_train_X_class, data_train_y_class, data_test_X_class, data_test_y_class = organize_train_test_class(
            list_ckt_names_class, test_set, train_data_label_path_class)
        # Fit model to classification data
        clf = gb.fit(data_train_X_class, data_train_y_class)
        mean_accuracy_train = clf.score(data_train_X_class, data_train_y_class)
        mean_accuracy_test = clf.score(data_test_X_class, data_test_y_class)
        predicted_class_train = gb.predict(data_train_X_class)
        predicted_class_test = gb.predict(data_test_X_class)
        # accuracy_score_value = accuracy_score(data_test_y_class, predicted_class)
        gb_confusion_matrix_train = confusion_matrix(data_train_y_class, predicted_class_train)
        gb_confusion_matrix_test = confusion_matrix(data_test_y_class, predicted_class_test)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(data_train_y_class, predicted_class_train).ravel()
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(data_test_y_class, predicted_class_test).ravel()

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

        # Matthews Correlation Coefficient
        mcc_train = matthews_corrcoef(data_train_y_class, predicted_class_train)
        mcc_test = matthews_corrcoef(data_test_y_class, predicted_class_test)

        # Save the model
        # save_file_name='gb' + str(decision_point) + '.pkl'
        # save_file_compressed_name = 'gb_compressed' + str(decision_point) + '.pkl'
        # joblib.dump(gb, save_file_name)
        # joblib.dump(gb, save_file_compressed_name, compress=1)
        # final_avg_score_for_k_fold_cross_validation.append(accuracy_score_value)
        data_row = [decision_point, max_tree_depth, no_of_trees, i, mean_accuracy_train, mean_accuracy_test,
                    tp_train, tn_train, fp_train, fn_train, tp_test, tn_test, fp_test, fn_test, mcc_train, mcc_test]
        writer.writerow(data_row)

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
    data_row = [decision_point, max_tree_depth, no_of_trees, "cumulative",mean_accuracy_train_final, mean_accuracy_test_final, tp_train_sum, tn_train_sum, fp_train_sum, fn_train_sum, tp_test_sum, tn_test_sum, fp_test_sum, fn_test_sum, mcc_train_final,mcc_test_final]
    writer.writerow(data_row)

# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument('--decision_point', dest='decision_point', type=str, help='Add decision_point')
args = parser.parse_args()
decision_point = args.decision_point

print('\nDecision point : ', decision_point)
print('Max Depth values :', max_tree_depth_list)
print('No: of trees (n_estimators) : ', no_of_trees_list)
print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
print('--------------------------------------------------------------------------------------')
print('\nLoad Regression Data : Train')
(data_train_X_reg, data_train_y_reg) = load_train_data_reg(train_data_file_path_reg, train_label_file_path_reg)
columns_X = data_train_X_reg.shape[1]  # no: of features

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
filename = 'results_new_' + str(decision_point) + '.csv'
with open(filename, 'w', newline='') as fileptr:
    writer = csv.writer(fileptr)
    writer.writerow(header)

    for m in max_tree_depth_list:
        max_tree_depth = int(m)
        for n in no_of_trees_list:
            no_of_trees = int(n)
            print('\nmax_depth : ', max_tree_depth)
            print('No: of trees (n_estimators) : ', no_of_trees)
            print('Start Time : ', time.strftime("%H:%M:%S", time.localtime()))

            print('  Load Classification Model')
            gb = GradientBoostingClassifier(random_state=0, max_depth=max_tree_depth, n_estimators=no_of_trees)
 
            print('  Perform '+str(k_fold_cross_validation)+'-fold cross validation and record metrics')
            perform_k_fold_cross_validation(list_ckt_names_class, k_fold_cross_validation, test_ckts_array, writer, gb)
            print('End Time : ', time.strftime("%H:%M:%S", time.localtime()))

fileptr.close()
print('\n--------------------------------------------------------------------------------------')
print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))

