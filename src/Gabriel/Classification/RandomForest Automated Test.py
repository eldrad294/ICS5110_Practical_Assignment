# Module imports
import seaborn as sns
from pandas import DataFrame, concat, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#
path = 'EEGEyeState_Training.arff.csv'
path_test = 'EEGEyeState_Testing.arff.csv'
lag = 0
df = pd.read_csv(path)
df_values = df.values
df_values_count = len(df_values)
df_headers = df.columns
df_header_count = len(df_headers)
print('DF size before outlier removal: ' + str(df_values_count))
#
# Removing outliers
def get_outlier_indexes(df, df_col_count, outlier_value=5000):
    """ Takes a dataframe and returns the position of any outliers """
    outLierIndexes = set()
    upperLimit = outlier_value
    for x in range(df_col_count):
        df = np.array(df)
        outLiers = np.where(df[:, x] > upperLimit)[0]
        if len(outLiers) > 0:
            [(outLierIndexes.add(outLiers[i])) for i in range(len(outLiers))]
    #
    outLierIndexes = list(outLierIndexes)
    outLierIndexes.sort()
    return outLierIndexes
#
outlier_indexes = get_outlier_indexes(df_values, df_header_count)
print('Extreme outliers\nTotal:   ', len(outlier_indexes), \
    '\nIndexes: ', outlier_indexes)
[(print(DataFrame(df).iloc[[outlier]])) for outlier in outlier_indexes]
df_values_pruned = np.delete(df_values, outlier_indexes, 0)
df_values_pruned_count = len(df_values_pruned)
df_pruned = pd.DataFrame(data=df_values_pruned, columns=df_headers)
#
print('DF size after outlier removal: ' + str(df_values_pruned_count))
#
# Applying lag to timeseries data set, by shifting degree of n
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#
# N-Step Univariate Forecasting Shift
df_pruned_shifted = series_to_supervised(data=df_pruned, n_in=lag, n_out=1, dropnan=True)
#
# Removing any lag variables of var15(t-lag) (label)
if lag > 0:
    for i in range(1,lag+1):
        df_pruned_shifted = df_pruned_shifted.drop('var15(t-' + str(i) + ')', 1)
df_pruned_shifted_headers = df_pruned_shifted.columns
df_pruned_shifted_header_count = len(df_pruned_shifted_headers)
#
# Importing Test File
df = pd.read_csv(path_test)
df_values = df.values
df_values_count = len(df_values)
df_headers = df.columns
df_header_count = len(df_headers)
#
outlier_indexes = get_outlier_indexes(df_values, df_header_count)
df_values_pruned = np.delete(df_values, outlier_indexes, 0)
df_values_pruned_count = len(df_values_pruned)
df_pruned = pd.DataFrame(data=df_values_pruned, columns=df_headers)
#
print('DF size after outlier removal: ' + str(df_values_pruned_count))
#
# N-Step Univariate Forecasting Shift
df_pruned_shifted2 = series_to_supervised(data=df_pruned, n_in=lag, n_out=1, dropnan=True)
#
# Removing any lag variables of var15(t-lag) (label)
if lag > 0:
    for i in range(1,lag+1):
        df_pruned_shifted2 = df_pruned_shifted2.drop('var15(t-' + str(i) + ')', 1)
df_pruned_shifted_headers2 = df_pruned_shifted2.columns
df_pruned_shifted_header_count2 = len(df_pruned_shifted_headers2)
#
# Cross-validating of dataset: Splitting dataset into sub samples for training and validation purposes
from sklearn.model_selection import train_test_split
df_pruned_shifted_Y2 = df_pruned_shifted2['var15(t)']
df_pruned_shifted_X2 = df_pruned_shifted2.drop('var15(t)', 1)
df_pruned_shifted_X2 = scaler.fit_transform(df_pruned_shifted_X2)
#
# Parameters
criterion = ['gini', 'entropy']
n_estimators = [10,100,1000, 2500, 5000, 7500, 10000]
max_features = [14, "auto","sqrt","log2"]
test_size = [.2,.3,.4]
#
from sklearn.ensemble import RandomForestClassifier
from src.statistics.scoring_functions import Scoring_Functions
from sklearn.model_selection import train_test_split
df_pruned_shifted_Y = df_pruned_shifted['var15(t)']
df_pruned_shifted_X = df_pruned_shifted.drop('var15(t)', 1)
count = -1
for c in criterion:
    for n in n_estimators:
        for m in max_features:
            for ts in test_size:
                X_train, X_test, y_train, y_test = train_test_split(df_pruned_shifted_X, df_pruned_shifted_Y,
                                                                    test_size=ts, random_state=0)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
                #
                clf = RandomForestClassifier(n_estimators=n, max_features=m, n_jobs=6, criterion=c)
                clf.fit(X_train, y_train)
                #
                # Make predictions for validation data and evaluate
                pred_y = clf.predict(X_test)
                #
                # Make predictions for testing data and evaluate
                pred_y2 = clf.predict(df_pruned_shifted_X2)
                #
                # Testing Classifier Accuracy on Verification Dataset
                sf2 = Scoring_Functions(y_pred=pred_y2, y_true=df_pruned_shifted_Y2)
                #
                # Testing Classifier Accuracy on Verification Dataset
                sf = Scoring_Functions(y_pred=pred_y, y_true=y_test)
                count += 1
                if sf2.accuracy() > 60 and sf2.f_measure() > 60:
                    print("(" + str(count) + ") Criterion: " + str(c) + "\nn_estimators: " + str(n) + "\nmax_features: " + str(m) + "\ntest_size: " + str(ts) + "\n------------")
                    print("Verification Sample:")
                    print(sf.scoring_results())
                    print("------------")
                    print("Test Sample:")
                    print(sf2.scoring_results())
                    print("----------------------------------------------------------------------")
