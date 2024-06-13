##########################################################################################################
# This script is used for testing the simple classifier models :
# gaussian_naive_bayes_with_balanced_class_priors, k_nearest_neighbors,
# linear_support_vector_machine_using_hinge_loss, logistic_regression, and simple_decision_tree_depth_6
# Please change the following variables according to the model : model_file_suffix, model_path_class
##########################################################################################################

# Import packages

import time
import joblib
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

# Declare Variables
header=["Decision Point", "Accuracy", "True Positive", "True Negative","False Positive", "False Negative", "MCC"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
model_file_suffix='-1_knn'
#model_file_suffix='-1_gnb_priors'
#model_file_suffix='-1_sgd_linear_svm'
#model_file_suffix = '-1_logistic'
#model_file_suffix='-1_decision_tree_d6'


# Specify File Paths
data_label_path_class = 'project_data/ttnkoi_class_50-1000_by_name/'
model_path_class = 'ML_MODELS/ml_models_knn/'
#model_path_class = 'ML_MODELS/ml_models_gnb_priors/'
#model_path_class = 'ML_MODELS/ml_models_svm/'
#model_path_class = 'ML_MODELS/ml_models_logistic/'
#model_path_class = 'ML_MODELS/ml_models_dt_6/'
decision_points_list = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650','700','750','800','850','900','950','1000']
#decision_points_list = ['100', '200', '300', '400', '500', '600', '700', '800',]
# ===================== Functions =================================================================

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

    # Concatenate info in csv files
    columns_X = np.genfromtxt(data_files_list[0], delimiter=',', dtype=float).shape[1]
    data_train_X = np.empty((0, columns_X))
    data_train_y = np.empty(0)

    for file in data_files_list:
        data_train_X_temp = np.genfromtxt(file, delimiter=',', dtype=float)
        data_train_X = np.concatenate((data_train_X, data_train_X_temp), axis=0)

    for file in label_files_list:
        data_train_y_temp = np.genfromtxt(file, delimiter=',', dtype=int, usecols=0)
        data_train_y = np.concatenate((data_train_y, data_train_y_temp))

    return data_train_X, data_train_y


# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
for d in decision_points_list:
    decision_point = int(d)
    print('\nDecision point : ', decision_point)
    model_file_name_class = 'mvto-' + str(decision_point) + model_file_suffix
    model_file_path_class = str(np.char.add(model_path_class, model_file_name_class))
    print('Model name : ', model_file_path_class)

    print('Load Classification Model')
    model_load_var_class = joblib.load(model_file_path_class)

    print('Load Classification Data')
    data_file_class = '-' + str(decision_point) + data_file_suffix
    label_file_class = '-' + str(decision_point) + label_file_suffix
    data_test_X_class, data_test_y_class = capture_test_data_label_class(data_label_path_class, data_file_class, label_file_class)

    print('Predict class')
    class_predicted = model_load_var_class.predict(data_test_X_class).round()

    #Calculate Metrics
    print('Calculate metrics')
    mean_accuracy = accuracy_score(data_test_y_class, class_predicted)
    tn, fp, fn, tp = confusion_matrix(data_test_y_class, class_predicted).ravel()
    mcc = matthews_corrcoef(data_test_y_class, class_predicted)
    print(header)
    print(str(decision_point) +','+ str(mean_accuracy) +','+ str(tp) +','+ str(tn) +','+ str(fp) +','+ str(fn) +','+ str(mcc))

print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))


