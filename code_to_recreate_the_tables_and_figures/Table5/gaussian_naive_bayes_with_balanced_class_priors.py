#################################################################################################
# This script trains Gaussian Naive Bayes model with balanced class priors and saves the model
#################################################################################################

# Import packages

import time
import joblib
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB


# Declare Variables
header=["Decision Point", "Classifier", "Accuracy (Train)", "True Positive (Train)",
        "True Negative(Train)","False Positive (Train)", "False Negative (Train)", "MCC (Train)"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
k_fold_cross_validation = int(5) ## 5 fold cross validation
decision_points_list = ['50','100', '150', '200', '250', '300', '350', '400', '450', '500',
                        '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000']

# Specify File Paths
train_data_file_path_reg = 'project_data/routing_data_train/mvto-1000-1_train_X.csv'
train_label_file_path_reg = 'project_data/routing_data_train/mvto-1000-1_train_y.csv'
train_data_label_path_class = 'project_data/circuit_split_train_data_for_cv/'
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
    return list_ckt_names_class


def organize_train_data(list_ckt_names_class,train_data_label_path_class):
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

    return data_train_X, data_train_y



# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

for d in decision_points_list:
    decision_point = int(d)
    print('\nDecision point : ', decision_point)
    print('\nStart Time : ', time.strftime("%H:%M:%S", time.localtime()))
    print('--------------------------------------------------------------------------------------')

    print('\nCapture Binary Classification Data')
    train_data_file_class = '-' + str(decision_point) + data_file_suffix
    train_label_file_class = '-' + str(decision_point) + label_file_suffix
    list_ckt_names_class = capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class)
    list_ckt_names_class.sort()
    #print(list_ckt_names_class)
        
    clf = GaussianNB(priors=[0.5, 0.5])
    print('  Load Classification Model : ', clf)
    print('  Fit model to classification data')
    # Fit model to classification data
    data_train_X_class, data_train_y_class = organize_train_data(list_ckt_names_class, train_data_label_path_class)
    clf.fit(data_train_X_class, data_train_y_class)

    #Calculate Metrics
    print('   Calculate metrics for Train data')
    mean_accuracy_train = clf.score(data_train_X_class, data_train_y_class)
    predicted_class_train = clf.predict(data_train_X_class)
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(data_train_y_class, predicted_class_train).ravel()
    mcc_train = matthews_corrcoef(data_train_y_class, predicted_class_train)
    print(header)
    print(str(decision_point) +','+ 'classifier_gnb_priors'+ ','+ str(mean_accuracy_train) +','+ str(tp_train) +','+ str(tn_train) +','+ str(fp_train) +','+ str(fn_train) +','+ str(mcc_train))
    print('    Save the model : '+'mvto-' + str(decision_point) + '-1_gnb_priors')
    save_file_name = 'mvto-' + str(decision_point) + '-1_gnb_priors'
    joblib.dump(clf, save_file_name)

print('\n--------------------------------------------------------------------------------------')
print('\nEnd Time : ', time.strftime("%H:%M:%S", time.localtime()))


