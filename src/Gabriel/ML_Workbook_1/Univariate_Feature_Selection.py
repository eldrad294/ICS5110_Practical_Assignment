import pandas as pd
from pandas import concat, DataFrame
import numpy as np
#
path = 'EEGEyeState.csv'
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
df_values_pruned = np.delete(df_values, outlier_indexes, 0)
df_values_pruned_count = len(df_values_pruned)
df_pruned = pd.DataFrame(data=df_values_pruned, columns=df_headers)
print('DF size after outlier removal: ' + str(df_values_pruned_count))
#
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
lag = 0
df_pruned_shifted = series_to_supervised(data=df_pruned, n_in=lag, n_out=1, dropnan=True)
#
# Removing any lag variables of var15(t-lag) (label)
if lag > 0:
    for i in range(1,lag+1):
        df_pruned_shifted = df_pruned_shifted.drop('var15(t-' + str(i) + ')', 1)
df_pruned_shifted_headers = df_pruned_shifted.columns
df_pruned_shifted_header_count = len(df_pruned_shifted_headers)
#
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
from sklearn.metrics import mutual_info_score
import operator
mi_scores = {}
for column in df_pruned_shifted_headers:
    if column != 'var15(t)':
        mi_scores[column] = mutual_info_score(df_pruned_shifted[column], df_pruned_shifted['var15(t)'])
print('Mutual Information: ')
print(sorted(mi_scores.items(), key=operator.itemgetter(1), reverse=True))
#
# Feature Selection using a wrapper (RandomForests)
# https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
df_pruned_shifted_Y = df_pruned_shifted['var15(t)']
df_pruned_shifted_X = df_pruned_shifted.drop('var15(t)', 1)
#
rf.fit(df_pruned_shifted_X, df_pruned_shifted_Y)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = df_pruned_shifted_X.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
print('Random Forest Classification For Feature Selection: ')
print(feature_importances)
#
# Plot the feature importance of the forest
outward_path = 'D:\\Projects\\ICS5110_Practical_Assignment\\src\\Gabriel\\ML_Workbook_1\\Findings\\Plots\\Lag' + str(lag) + '\\'
import matplotlib.pyplot as plt
fig = plt.figure()
plt.title("Feature importance")
objects = list(feature_importances.axes[0])
y_pos = np.arange(len(objects))
feature_importance = np.array(feature_importances)
plt.bar(y_pos, feature_importance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
fig.savefig(outward_path + 'RandomForestClassifier_FeatureImportance.png')
#
# Plot the feature importance of the SVM
# https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
from sklearn import svm
#
svm = svm.SVC(kernel='linear')
svm.fit(df_pruned_shifted_X, df_pruned_shifted_Y)
df_pruned_shifted_X_columns = df_pruned_shifted_X.columns
feature_importances = pd.DataFrame(svm.coef_[0],
                                   index = df_pruned_shifted_X.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
print('SVM Classification For Feature Selection: ')
print(feature_importances)
#
# Plot the feature importance of the SVM
import matplotlib.pyplot as plt
fig = plt.figure()
plt.title("Feature importance")
objects = list(feature_importances.axes[0])
y_pos = np.arange(len(objects))
feature_importance = np.array(feature_importances)
plt.bar(y_pos, feature_importance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
fig.savefig(outward_path + 'SVM_FeatureImportance.png')
#
# Save feature data as separate line graphs
for column in df_pruned_shifted_headers:
    fig = plt.figure()
    if column != 'var15(t)':
        plt.plot(df_pruned_shifted[column])
        plt.ylabel(column)
        fig.savefig(outward_path + column + '.png')
#
# Cross-validating of dataset: Splitting dataset into sub samples for training and testing purposes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_pruned_shifted_X[['var7(t-3)']], df_pruned_shifted_Y, test_size=0.2, random_state=0)
#
#Training SVM
from sklearn import svm
kernel = 'rbf'
C=1
gamma = 'auto'
degree = 3
clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
clf.fit(X_train, y_train)
print(clf)
#
# Test Testing Sub sample
y_pred = []
X_test = np.array(X_test)
for i in range(len(y_test)):
    y_pred.append(clf.predict([X_test[i]]))
#
# Testing Classifier Accuracy
from src.statistics.scoring_functions import Scoring_Functions
sf = Scoring_Functions(y_pred=y_pred, y_true=y_test)
print(sf.scoring_results())
