#############################################################################################################
# This script trains GradientBoosting Classifier models for given hyperparameter values
# and captures the calibration metrics and performance scores
##############################################################################################################

# Import packages

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import joblib



# Declare Variables
header=["Decision Threshold", "Max depth", "No of Trees", "True Positive (Train)",
        "True Negative(Train)","False Positive (Train)", "False Negative (Train)", "True Positive (Test)",
        "True Negative(Test)","False Positive (Test)", "False Negative (Test)", "Accuracy (Train)", "MCC (Train)", "ECE(Train)", "Brier_score (Train)", "NLL (train)", "Accuracy (Test)", "MCC (Test)", "ECE(Test)", "Brier_score (Test)", "NLL (Test)"]

data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'
range_for_visualization = int(5)   # Range of probability values : [0%-5%][5%-10%] ...
max_tree_depth_list = ['6']
no_of_trees_list = ['30']

# Specify File Paths

train_data_label_path_class = 'G:/ProjectData/circuit_split_train_data_for_cv/'
test_data_label_path_class = 'G:/ProjectData/ttnkoi_class_50-1000_by_name/'
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
    #print('Circuit Names :')
    #pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(list_ckt_names_class)
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


def visualize_routable_percentage(class_proba_array, data_y_class, color, decision_point, range_for_visualization, plot_title):
    """
    Visualize % Confidence (Predicted) Vs % Routable (Actual) 
    @param class_proba_array
    @param data_y_class
    @param color
    @param decision_point
    @param range_for_visualization
    @param plot_title
    @return :
    """
    proba_percent = np.vectorize(lambda x: x * 100)
    class_probabilities_percent_rnd = np.rint(proba_percent(class_proba_array))
    proba1 = class_probabilities_percent_rnd[:, [1]]

    concat_prediction_and_ground_truth = np.concatenate((proba1, data_y_class[:, None]), axis=1)

    array_len = int(100 / range_for_visualization)
    confidence_bins = np.arange(0, 99, range_for_visualization)
    predictions_count_in_given_range_array = np.zeros(array_len)
    routable_count_array = np.zeros(array_len)
    routable_percent_array = np.zeros(array_len)
    display_array = []

    for i in range(0, array_len):
        range_lower = (range_for_visualization * i)
        range_upper = (range_for_visualization * (i + 1)) + 1
        predictions_in_given_range = concat_prediction_and_ground_truth[np.where((concat_prediction_and_ground_truth[:, 0] > range_lower) * (concat_prediction_and_ground_truth[:, 0] < range_upper))]
        total_count = predictions_in_given_range.shape[0]
        if total_count == 0:
            total_count = 1
        predictions_count_in_given_range_array[i] = total_count
        routable_list = predictions_in_given_range[predictions_in_given_range[:,1] == 1]
        routable_count = routable_list.shape[0]
        routable_count_array[i] = routable_count
        routable_percentage = (routable_count/total_count) * 100
        routable_percent_array[i] = routable_percentage
        display_string=format(total_count,"0.0e")
        display_array.append(display_string)


    #print('Predictions in each range')
    #print(confidence_bins)
    #print(predictions_count_in_given_range_array)
    #print('Routable counts')
    #print(routable_count_array)
    #print('Routable percent')
    #print(routable_percent_array)
    bar_width = 3.5
    plt.style.use('ggplot')
    #plt.bar(confidence_bins, routable_percent_array, color=color,)
    fig, ax = plt.subplots()
    rects = ax.bar(confidence_bins, routable_percent_array, width=bar_width, color=color, data=display_array)
    ax.bar_label(rects, ('\n'.join(i) for i in display_array), label_type='center', fontsize=5)
    # Plot the ideal straight line , y=x
    ax.axline((0, 0), slope=1)
    plt.xlabel('% Confidence (Predicted)')
    plt.ylabel('% Routable (Actual)')
    plt.title(plot_title + str(decision_point))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig(pp, format='pdf', dpi=100)
    #plt.show()
    plt.clf()

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

# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++
now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--decision_point', dest='decision_point', type=str, help='Add decision_point')
args = parser.parse_args()
decision_point = args.decision_point
decision_point = 300
pdf_name = 'plot_gradb_separate_train_test_'+str(decision_point)+'.pdf'
pp = PdfPages(pdf_name)

print('\nDecision point : ', decision_point)
print('Max Depth values :', max_tree_depth_list)
print('No: of trees (n_estimators) : ', no_of_trees_list)
#print('\nProgram Start Time : ', time.strftime("%H:%M:%S", time.localtime()))
print('\nProgram Start Time : ',now.strftime("%Y-%m-%d %H:%M:%S"))
print('--------------------------------------------------------------------------------------')

print('\nCapture Binary Classification Data')
train_data_file_class = '-' + str(decision_point) + data_file_suffix
train_label_file_class = '-' + str(decision_point) + label_file_suffix

#Train Data
list_train_ckt_names_class = capture_ckt_info_class(train_data_label_path_class, train_data_file_class, train_label_file_class)
list_train_ckt_names_class.sort()
print('      Circuits in Train set : ', len(list_train_ckt_names_class))
print(list_train_ckt_names_class)
data_train_X_class, data_train_y_class = organize_train_data(list_train_ckt_names_class, train_data_label_path_class, train_data_file_class, train_label_file_class)

#Test Data
list_test_ckt_names_class = capture_ckt_info_class(test_data_label_path_class, train_data_file_class, train_label_file_class)
list_test_ckt_names_class.sort()
print('      Circuits in Test set : ', len(list_test_ckt_names_class))
print(list_test_ckt_names_class)
data_test_X_class, data_test_y_class = organize_train_data(list_test_ckt_names_class, test_data_label_path_class, train_data_file_class, train_label_file_class)

for m in max_tree_depth_list:
    max_tree_depth = int(m)
    for n in no_of_trees_list:
        no_of_trees = int(n)
        print('\nmax_depth : ', max_tree_depth)
        print('No: of trees (n_estimators) : ', no_of_trees)
        print('  Load Classification Model')
        gb = GradientBoostingClassifier(random_state=0, max_depth=max_tree_depth, n_estimators=no_of_trees)

        print('  Fit model to Train data')
        # Fit model to classification data
        clf = gb.fit(data_train_X_class, data_train_y_class)
        save_file_name = 'mvto-' + str(decision_point) + '-1_gradb_class_' + str(max_tree_depth) +'d' + str(no_of_trees) + 't'
        joblib.dump(clf, save_file_name)

        # Inference : Train
        print('  Inference/Predict')
        class_proba_train_array = clf.predict_proba(data_train_X_class)
        class_proba_test_array = clf.predict_proba(data_test_X_class)
        class_predicted_train = clf.predict(data_train_X_class).round()
        class_predicted_test = clf.predict(data_test_X_class).round()

        # Visualize routable percentage
        print("Visualize Train Data")
        visualize_routable_percentage(class_proba_train_array, data_train_y_class, 'orange', decision_point, range_for_visualization, 'Train Data : Decision Threshold = ')

        print("Visualize Test Data")
        visualize_routable_percentage(class_proba_test_array, data_test_y_class, 'cyan', decision_point, range_for_visualization, 'Test Data : Decision Threshold = ')

        print('Calculate metrics')
        mean_accuracy_train = accuracy_score(data_train_y_class, class_predicted_train)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(data_train_y_class, class_predicted_train).ravel()
        mcc_train = matthews_corrcoef(data_train_y_class, class_predicted_train)

        mean_accuracy_test = accuracy_score(data_test_y_class, class_predicted_test)
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(data_test_y_class, class_predicted_test).ravel()
        mcc_test = matthews_corrcoef(data_test_y_class, class_predicted_test)

        print("Metrics for Train Data")

        ece_train = expected_calibration_error_tempscale(data_train_y_class, class_proba_train_array, num_bins=20)
        brier_score_train, nll_train = get_calibration_metrics(class_proba_train_array,data_train_y_class)
        print("\nMetrics for Test Data")
        ece_test = expected_calibration_error_tempscale(data_test_y_class, class_proba_test_array, num_bins=20)
        brier_score_test, nll_test = get_calibration_metrics(class_proba_test_array, data_test_y_class)

        print(header)
        print(str(decision_point) + ',' + str(max_tree_depth) + ',' + str(no_of_trees) + ',' + str(
            tp_train) + ',' + str(tn_train) + ',' + str(fp_train) + ',' + str(fn_train) + ',' + str(
            tp_test) + ',' + str(tn_test) + ',' + str(fp_test) + ',' + str(fn_test) + ',' + str(
            mean_accuracy_train) + ',' + str(
            mcc_train) + ',' + str(ece_train) + ',' + str(brier_score_train) + ',' + str(nll_train) + ',' + str(
            mean_accuracy_test) + ',' + str(mcc_test) + ',' + str(
            ece_test) + ',' + str(brier_score_test) + ',' + str(nll_test))

        print('\n--------------------------------------------------------------------------------------')
    #print('\nProgram End Time : ', time.strftime("%H:%M:%S", time.localtime()))
    now = datetime.datetime.now()
    print('\nProgram End Time : ',now.strftime("%Y-%m-%d %H:%M:%S"))
pp.close()
