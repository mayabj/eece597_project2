####################################################################################################################
#
# This script is used to find optimum depth for small decision trees
# The parameter 'max_depth' is varied and plots for class probability and absolute number of iterations are captured.
# Expected nature of curves in the ideal scenario:
#      - probability plots should not show overfitting behavior (2 level clipping)
#      - plots for absolute number of iterations should cover a wide range without gaps
# The metrics Mean Absolute Error (MAE) and r2 score are reported in results.csv file
# This analysis is performed for various decision thresholds and based on the curves, the optimum value of max_depth is decided.

###################################################################################################################

# Import packages

import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy_indexed as npi
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier


# Declare Variables
#class_models_array = np.array(['75','150', '250', '400'], dtype=object)
class_models_array = np.array(['150'], dtype=object)
file_prefix='mvto-'
data_file_suffix='-1_train_X.csv'
label_file_suffix='-1_train_y.csv'

# Specify File Paths
train_data_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_X.csv'
train_label_file_path_reg = 'G:/ProjectData/routing_data_train/mvto-1000-1_train_y.csv'
file_path_class='G:/ProjectData/class_train_data_75_150_250_400_1000/'

# ===================== Functions =================================================================

def create_y_abs_array(number_of_runs, x_data_array, y_label_array):
    """
        # Create an array to capture information on y_absolute for each routing run
        # Eg: if y_label_array contains the following
        # row 0 : 5
        # row 1 : 4
        # row 2 : 3
        # row 3 : 2
        # row 4 : 1
        # row 5 : 100
        # Then, the array y_abs_info should be
           # row 0 : 0, 6
           # row 1 : 5, 101
        # The array y_abs should be
           # row 0 : 6
           # row 1 : 6
           # row 2 : 6
           # row 3 : 6
           # row 4 : 6
           # row 5 : 101

        @param number_of_runs:
        @param x_data_array:
        @param y_label_array:
        @return:
    """
    y_abs_info = np.empty((number_of_runs + 1, 2))
    y_abs_info[0][0] = 0
    y_abs_info[0][1] = y_label_array[0] + 1

    # Get all the y_abs and corresponding index values from y_label_array
    arr_1 = np.full(number_of_runs, 1)
    indices_for_1 = np.where(np.in1d(y_label_array, 1)) + arr_1
    indices_for_1 = indices_for_1.transpose()

    for i in range(0, (number_of_runs - 1)):
        y_abs_info[i + 1][0] = indices_for_1[i]
        y_abs_info[i + 1][1] = y_label_array[indices_for_1[i]] + 1
    y_abs_info[number_of_runs][0] = x_data_array.shape[0]
    y_abs_info[number_of_runs][1] = 0

    y_abs_info = y_abs_info.astype(int)

    # Create a new Y_abs label array using y_abs_info
    y_abs = np.empty(x_data_array.shape[0])
    for i in range(0, number_of_runs):
        col_start = y_abs_info[i][0]
        col_end = y_abs_info[i + 1][0]
        y_abs_val = y_abs_info[i][1]
        y_abs[col_start:col_end] = y_abs_val

    return y_abs_info, y_abs


def load_train_data(train_data_file_path,train_label_file_path):
    """
    Reads in Train Data
    @param train_data_file_path:
    @param train_label_file_path:
    @return : data_train_X,data_train_y,y_train_abs_array
    """
    data_train_X = np.genfromtxt(train_data_file_path, delimiter=',', dtype=float)
    data_train_y = np.genfromtxt(train_label_file_path, delimiter=',', dtype=int, usecols=0)

    # Find how many routing runs are captured in the Y_labels.csv
    # Number of routing runs = number of 1s
    number_of_runs_train = list(data_train_y.flatten()).count(1)
    print("                Number of routing runs in train data = ", number_of_runs_train)

    #print("   Capturing Y_absolute labels for train ...")
    (y_train_abs_array_info, y_train_abs_array) = create_y_abs_array(number_of_runs_train, data_train_X, data_train_y)
    y_train_abs_array = y_train_abs_array[:, None]  # To adjust the dimensions
    return data_train_X,data_train_y,y_train_abs_array



def logistic(x, L, x0, k, b):
    """
    @param x: input
    @param L: curve’s maximum y value
    @param x0: midpoint of the sigmoid
    @param k: logistic growth rate or steepness of the curve
    @param b : curve's min y value
    @return:
    """
    # Sigmoid curve : L=1, x0=0, k=1
    return (((L - b) / (1.0 + np.exp(-k * (x - x0)))) + b)
    #return L / (1.0 + np.exp(-k * (x - x0)))


def calculate_r2_score(function_name, x_array, y_array, data_type, *popt):
    """
       # Takes x_array, y_array & *popt and calculates goodness of curve (Rsquared score)
       @param function_name
       @param x_array:
       @param y_array:
       @param data_type:
       @param *popt
    """
    predicted_value = function_name(x_array, *popt)
    return r2_score(y_array, predicted_value)

def show_plot(xlabel, ylabel, title_text):
    """
       # Show plot
       @param xlabel:
       @param ylabel:
       @param title_text:
    """
    plt.style.use('ggplot')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend(loc='upper right')
    plt.title(title_text)
    plt.show()
    plt.close()

def inverse_logistic(confidence_val, L, x0, k, b):
    """
    @param confidence_val: confidence values
    @param L: Logistic curve’s maximum y value
    @param x0: midpoint of the sigmoid
    @param k: growth rate or steepness of the logistic curve
    @param b : logistic curve's min y value
    @return:
    """
    # y_predict equation is derived taking inverse of logistic equation
    # y_predict = (logarithm((confidence_val - b) / (L - confidence_val)) + k * x0) / k

    # Conditions to avoid logarithm of negative number and logarithm of zero
    # If confidence_value <= b, y_predict = 1000 , maximum number of iterations
    # If confidence_value >= L, y_predict = 2 , minimum number of iterations
    val1 = float(1000.0)
    val2 = float(2.0)

    # Following long expression for y_predict gives RuntimeWarning: invalid value encountered in log.
    # y_predict = np.where((confidence_val <= b), val1, np.where((confidence_val >= L), val2, (((np.log((confidence_val - b) / (L - confidence_val))) + k * x0) / k)))
    # Hence, splitting the equation to multiple terms

    # If 1000 < confidence_val < b , then logarithm of expr1 can not be evaluated.
    # Hence, assign 1.0, so that term1 = log (1) = 0
    expr1 = np.where(((confidence_val <= b) | (confidence_val >= L)), 1.0, (confidence_val - b) / (L - confidence_val))
    term1 = np.log(expr1)
    term2 = k * x0
    y_predict = np.where((confidence_val <= b), val1, np.where((confidence_val >= L), val2, ((term1 + term2) / k)))
    #print(y_predict)
    # Replace all elements greater than 1000 to 1000 because maximum number of iterations = 1000
    # Replace all elements less than 2 to 2 because minimum number of iterations = 2
    y_predict[y_predict > val1] = val1
    y_predict[y_predict < val2] = val2
    return y_predict

def predict_using_logistic_curve_fitting_v1(y_train_abs_array, class_proba_one_train_array,model_info):
    """
        Perform logistic curve fitting for Train data
        @param y_train_abs_array:
        @param class_proba_one_train_array:
        @param model_info:
        #@return:
     """
    logistic_fit_params_abs_array = np.empty((4,class_proba_one_train_array.shape[1]))  #logistic curve fitting has 4 parameters
    y_abs_pred_with_popt_abs_train = np.empty((class_proba_one_train_array.shape[0],class_proba_one_train_array.shape[1]))

    y_train_abs_array_float = list(map(float, y_train_abs_array))
    class_proba_one_train_array_float = list(map(float,class_proba_one_train_array))
    y_abs_uniq_val, mean_proba_for_y_abs_uniq_val = npi.group_by(y_train_abs_array_float).mean(class_proba_one_train_array_float)
    # Initial guess for L, x0 , k and b are placed in p_0
    # L: the curve’s maximum y value
    # x0: the midpoint of the sigmoid
    # k: logistic growth rate or steepness of the curve
    # b: the curve's minimum y value
    L_estimate = np.max(mean_proba_for_y_abs_uniq_val)
    x0_estimate = np.median(y_abs_uniq_val)
    k_estimate = -L_estimate / (np.max(y_abs_uniq_val)-np.min(y_abs_uniq_val))
    b_estimate = np.min(mean_proba_for_y_abs_uniq_val)
    p_0 = [L_estimate, x0_estimate, k_estimate, b_estimate]
    #print('Seed values are, p0:', p_0)

    #  Curve fitting with respect to y_abs
    # popt : stores the optimized parameters to the logistic function minimizing the loss
    # pcov : estimated covariance of popt
    # bounds : To constrain the optimization to a specific range
    # eg: Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
    # eg : popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    # Syntax for bounds => ([Lmin,x0min, kmin, bmin],[Lmax,x0max,kmax, bmax])
    # maxfev : max number of iterations
    # Optimization options : {lm, dogbox, trf} ; 'lm' works only for unconstrained problems, i.e. without bounds
    x0_bound_max = float(x0_estimate) + 200.0
    popt_abs, pcov_abs = curve_fit(logistic, y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, p_0, method='trf', maxfev=1000000, bounds=([0., 0., -0.5, 0.0], [100.0, x0_bound_max, 1.0, 10.0]))

    logistic_fit_params_abs_array[:,0] = popt_abs
    # Curve fitting plots
    plt.style.use('ggplot')
    proba_predict_train_abs = logistic(y_abs_uniq_val, *popt_abs)
    title_logistic_abs = str(model_info) + ' : Logistic curve fitting' + '\nAvg class probability (1) for each Yabs'
    xlabel = 'Yabs (Absolute no:of iterations)'
    ylabel = 'Avg class probability (1)'
    plt.plot(y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, 'o', color='green', label='proba_true')
    plt.plot(y_abs_uniq_val, proba_predict_train_abs, color='black', label='proba_predicted')
    plt.legend(loc='upper right')
    show_plot(xlabel, ylabel, title_logistic_abs)


    # Performance metrics for curve fitting with unique y_abs
    # For reliable results, the model should not be overparametrized;
    # redundant parameters can cause unreliable covariance matrices and, in some cases, poorer quality fits
    # Calculate the condition number of the covariance matrix to check whether the model may be overparameterized
    # Covariance matrices with large condition numbers may indicate that results are unreliable
    #print('     Condition number of covariance matrix pcov_abs:', np.linalg.cond(pcov_abs))
    # The diagonal elements of the covariance matrix, which is related to uncertainty of the fit, gives more information
    #print('     Diagonal values of covariance matrix pcov_abs:', np.diag(pcov_abs))
    #r2_score_fit_abs_array[i] = calculate_r2_score(logistic, y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, *popt_abs)
    r2_score_fit_abs_array = calculate_r2_score(logistic, y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, 'curve fitting', *popt_abs)

    # Plot inverse_logistic
    class_proba_inputs = np.arange(1, 101, 1)
    title_inv_logistic_abs = str(model_info) + ' : Inverse Logistic' + "\nClass probability Vs predicted no: of iterations"
    y_predict_abs = inverse_logistic(class_proba_inputs, *popt_abs)
    plot_inverse_curve(class_proba_inputs, y_predict_abs, 'purple', title_inv_logistic_abs)

    # Predict no:of iterations
    y_abs_pred_with_popt_abs_train[:, 0] = inverse_logistic(class_proba_one_train_array[:,0], *popt_abs)
    r2_score_predict_abs_train = calculate_r2_score(inverse_logistic, class_proba_one_train_array[:,0], y_train_abs_array, 'Train prediction', *popt_abs)
    # print("R2 score for Train prediction =", r2_score_predict_abs_train, file=file_pointer)

    # Calculate Mean Absolute Error and plot y_true vs y_pred
    xlabel_predict_class = 'Number of iterations, Yabs'
    ylabel_predict_class = 'Y_abs prediction from classification'
    title_pred_class_abs = str(model_info) + ' : Logistic curve fitting' + '\nYabs Vs Y_prediction : '
    mae_abs_train = plot_y_predict(y_train_abs_array_float, y_abs_pred_with_popt_abs_train[:, 0], 'magenta', title_pred_class_abs, ' Train', xlabel_predict_class, ylabel_predict_class)
    # print("Mean Absolute Error for Train Data =", mae_abs_train, file=file_pointer)
    return r2_score_predict_abs_train,mae_abs_train

def plot_y_predict(y_true_array, y_predict_array, color, title, data_type, xlabel, ylabel):
    """
    Plots y_true Vs y_predict
    @param y_true_array
    @param y_predict_array
    @param color
    @param title
    @param data_type
    @param xlabel
    @param ylabel
    """
    plt.plot(y_true_array, y_predict_array, 'o', color=color, label='_nolegend_')
    title_updated = str(title) + str(data_type) + " Data"
    show_plot(xlabel, ylabel, title_updated)

    #Calculate Mean Absolute Error
    mae = mean_absolute_error(y_true_array,y_predict_array)
    #print('       Mean Absolute Error for' + str(data_type) + ' Data = ', mae)

    return mae
def plot_inverse_curve(class_proba_array, y_predict_abs_array, color, title):
    """
       # Takes class_proba_array & y_predict_abs_array
       @param class_proba_array:
       @param y_predict_abs_array:
    """
    plt.scatter(class_proba_array, y_predict_abs_array, color=color, label='y_predict')
    plt.ylim(0, 1000)
    xlabel_inv_logistic = "Class probability"
    ylabel_inv_logistic = 'Yabs (Absolute no:of iterations)'
    show_plot(xlabel_inv_logistic, ylabel_inv_logistic, title)



def negative_exp(x, a, b, c):
    """
    @param x: input
    @param a: max value
    @param b : slope
    @param c: min value
    @return:
    """
    return a * np.exp(-b * x) + c


def inverse_exp(confidence_val, a, b, c):
    """
    @param confidence_val: confidence values
    @param a: curve’s maximum y value
    @param b: growth rate or steepness of the curve
    @param c: curve's min y value
    @return:
    """
    # y_predict equation is derived taking inverse of exponential equation
    # y_predict = (logarithm(a / (confidence_val - c))) / b

    # Conditions to avoid logarithm of negative number and logarithm of zero
    # If confidence_value <= c, y_predict = 1000 , maximum number of iterations
    # If confidence_value >= a, y_predict = 2 , minimum number of iterations
    val1 = float(1000.0)
    val2 = float(2.0)

    y_predict = np.where((confidence_val <= c), val1,
                         np.where((confidence_val >= a), val2, ((np.log(a / (confidence_val - c))) / b)))

    # Replace all elements greater than 1000 to 1000 because maximum number of iterations = 1000
    # Replace all elements less than 2 to 2 because minimum number of iterations = 2
    y_predict[y_predict > val1] = val1
    y_predict[y_predict < val2] = val2
    return y_predict


def predict_using_exponential_curve_fitting_v1(y_train_abs_array, class_proba_one_train_array, model_info):
    """
        Perform exponential curve fitting for Train data
        @param y_train_abs_array:
        @param class_proba_one_train_array:
        @param model_info:
        #@return:
    """
    exp_fit_params_array = np.empty((3, class_proba_one_train_array.shape[1]))  # exponential curve fitting has 3 parameters
    y_abs_pred_with_popt_train = np.empty((class_proba_one_train_array.shape[0],class_proba_one_train_array.shape[1]))
    y_train_abs_array_float = list(map(float, y_train_abs_array))
    class_proba_one_train_array_float = list(map(float,class_proba_one_train_array[:,0]))
    y_abs_uniq_val, mean_proba_for_y_abs_uniq_val = npi.group_by(y_train_abs_array_float).mean(class_proba_one_train_array_float)
    # Initial guess are placed in p_0
    a_estimate = np.max(mean_proba_for_y_abs_uniq_val)
    b_estimate = a_estimate / (np.max(y_abs_uniq_val)-np.min(y_abs_uniq_val))
    c_estimate = np.min(mean_proba_for_y_abs_uniq_val)
    p_0 = [a_estimate, b_estimate, c_estimate]

    #  Curve fitting with respect to y_abs
    popt, pcov = curve_fit(negative_exp, y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, p_0, maxfev=1000000)
    exp_fit_params_array[:, 0] = popt

    # Curve fitting plots
    plt.style.use('ggplot')
    proba_predict_train = negative_exp(y_abs_uniq_val, *popt)
    title_exp = str(model_info) + ' : Exp curve fitting' + '\nAvg class probability (1) for each Yabs'
    xlabel = 'Yabs (Absolute no:of iterations)'
    ylabel = 'Avg class probability (1)'
    plt.plot(y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, 'o', color='blue', label='proba_true')
    plt.plot(y_abs_uniq_val, proba_predict_train, color='black', label='proba_predicted')
    plt.legend(loc='upper right')
    show_plot(xlabel, ylabel, title_exp)

    # Performance metrics for curve fitting with unique y_abs
    r2_score_fit = calculate_r2_score(negative_exp, y_abs_uniq_val, mean_proba_for_y_abs_uniq_val, 'curve fitting',*popt)

    # Plot inverse_exp
    class_proba_inputs = np.arange(1, 101, 1)
    title_inv_exp = str(model_info) + ' : Inverse Exponential' + "\nClass probability Vs predicted no: of iterations"
    y_predict = inverse_exp(class_proba_inputs, *popt)
    plot_inverse_curve(class_proba_inputs, y_predict, 'red', title_inv_exp)

    # Predict no:of iterations
    y_abs_pred_with_popt_train[:, 0] = inverse_exp(class_proba_one_train_array[:, 0], *popt)
    r2_score_predict_abs_train = calculate_r2_score(inverse_exp, class_proba_one_train_array[:, 0], y_train_abs_array, 'Train prediction', *popt)

    # Calculate Mean Absolute Error and plot y_true vs y_pred
    xlabel_predict_class = 'Number of iterations, Yabs'
    ylabel_predict_class = 'Y_abs prediction from classification'
    title_pred_class = str(model_info) + ' : Exponential curve fitting' + '\nYabs Vs Y_prediction : '

    mae_abs_train = plot_y_predict(y_train_abs_array_float, y_abs_pred_with_popt_train[:, 0], 'orange', title_pred_class, ' Train', xlabel_predict_class, ylabel_predict_class)
    return r2_score_predict_abs_train, mae_abs_train


# =============================== End of functions ===================================================

# ++++++++++++++++++++++++++++++++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++

header=["Decision Point", "Max depth", "Max leaf nodes", "MAE(logistic)", "MAE(exponential)", "R2 score for prediction(logistic)", "R2 score for prediction(exponential)"]
max_depth_array=[9,8]

with open('../codes_save_Feb_2024/results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    for decision_point in class_models_array:
        print('\nDecision point :  ' + str(decision_point))
        print('--------------------------------------------')
        train_data_file_class=file_prefix + decision_point + data_file_suffix
        train_label_file_class=file_prefix + decision_point + label_file_suffix
        train_data_file_path_class = file_path_class + train_data_file_class
        train_label_file_path_class = file_path_class + train_label_file_class

        for max_depth_val in max_depth_array:
             print('\n  max_depth :  ' + str(max_depth_val))
             model_info = str(decision_point) + '_depth_' + str(max_depth_val)
             print('model info : ' , model_info)
             print('     STEP (1) : Load Classification Model')
             dtc = DecisionTreeClassifier(random_state = 0, max_depth=max_depth_val)

             print('     STEP (2) : Load Regression Data : Train')
             (data_train_X_reg, data_train_y_reg, y_train_abs_array) = load_train_data(train_data_file_path_reg, train_label_file_path_reg)

             print('     STEP (3) : Load Binary Classification Data : Train')
             (data_train_X_class, data_train_y_class, y_class_array) = load_train_data(train_data_file_path_class, train_label_file_path_class)

             print('     STEP (4) : Fit model to classification data')
             dtc.fit(data_train_X_class, data_train_y_class)
             max_depth = dtc.get_depth()
             max_leaf_nodes = dtc.get_n_leaves()

             print('     STEP (5) : Inference/Predict for regression dataset')
             class_proba_train_array = dtc.predict_proba(data_train_X_reg)

             proba_percent = np.vectorize(lambda x: x * 100)
             class_probabilities_train_percent_rnd = np.rint(proba_percent(class_proba_train_array))
             c1_train = class_probabilities_train_percent_rnd[:, [1]]

             print('     STEP (6) : Perform Logistic curve fitting for Train Data and use the resulting function to predict absolute no:of iterations')
             (r2_score_predict_abs_train_logistic,mae_abs_train_logistic)=predict_using_logistic_curve_fitting_v1(y_train_abs_array, c1_train, model_info)

             print('     STEP (7_gradb_calibrate_for_each_iter_separate) : Perform Exponential curve fitting for Train Data and use the resulting function to predict absolute no:of iterations')
             (r2_score_predict_abs_train_exp,mae_abs_train_exp)=predict_using_exponential_curve_fitting_v1(y_train_abs_array, c1_train, model_info)
             data_row=[decision_point,max_depth,max_leaf_nodes,mae_abs_train_logistic,mae_abs_train_exp,r2_score_predict_abs_train_logistic,r2_score_predict_abs_train_exp]
             writer.writerow(data_row)

file.close()



