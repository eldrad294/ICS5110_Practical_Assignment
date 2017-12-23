from pandas import DataFrame, concat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from src.statistics.scoring_functions import Scoring_Functions
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
def get_outlier_indexes(df, df_count, outlier_value=5000):
    """ Takes a dataframe and returns the position of any outliers """
    outLierIndexes = set()
    upperLimit = outlier_value
    for x in range(df_count):
        df = np.array(df)
        outLiers = np.where(df[:, x] > upperLimit)[0]
        if len(outLiers) > 0:
            [(outLierIndexes.add(outLiers[i])) for i in range(len(outLiers))]
    #
    outLierIndexes = list(outLierIndexes)
    outLierIndexes.sort()
    return outLierIndexes
#
path = 'EEGEyeState.csv'
df = pd.read_csv(path)
#
# N-Step Univariate Forecasting Shift
lag = 0
shifted_df = series_to_supervised(data=df, n_in=lag, n_out=1, dropnan=True)
#
# Taking only the top 5 relevant features (Lag 0 - 6)
important_columns = [['var7(t)','var6(t)','var2(t)','var13(t)','var14(t)','var15(t)'],
                     ['var7(t)','var7(t-1)','var6(t-1)','var6(t)','var2(t-1)','var15(t)'],
                     ['var7(t-2)','var7(t)','var6(t-2)','var7(t-1)','var6(t)','var15(t)'],
                     ['var7(t-3)','var(t)','var6(t-3)','var6(t-2)','var7(t-1)','var15(t)'],
                     ['var7(t-4)','var7(t)','var6(t-3)','var7(t-2)','var6(t-4)','var15(t)'],
                     ['var6(t-5)','var7(t)','var6(t)','var6(t-2)','var7(t-3)','var15(t)']]
shifted_df = shifted_df[important_columns[lag]]
#
shifted_df_headers = shifted_df.columns
shifted_feature_count = len(shifted_df_headers)
#print(shifted_feature_count)
#
# Converting shifted dataframe into separate feature and label dataframes
shifted_df_X = shifted_df.drop('var15(t)', 1)
for i in range(1,lag+1):
    column = 'var15(t-' + str(i) + ')'
    shifted_df_X = shifted_df_X.drop(column, 1)
shifted_columns = shifted_df_X.columns
shifted_df_X_count = len(shifted_columns)
shifted_df_X = shifted_df_X.values
shifted_df_y = shifted_df[['var15(t)']].values
#
# Removing Outliers
outLierIndexes = get_outlier_indexes(shifted_df_X, shifted_df_X_count, 5000)
shifted_df_X = np.delete(shifted_df_X, outLierIndexes, 0)
shifted_df_y = np.delete(shifted_df_y, outLierIndexes, 0)
#
print('Size of features: ' + str(len(shifted_df_X)) + '\nSize of labels: ' + str(len(shifted_df_y)))
#
# Cross-validating of dataset: Splitting dataset into sub samples for training and testing purposes
X_train, X_test, y_train, y_test = train_test_split(shifted_df_X, shifted_df_y, test_size=0.2, random_state=0)
#
# Feed through SVM classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf)
#
# Test Testing Sub sample
y_pred = []
for i in range(len(y_test)):
    y_pred.append(clf.predict([X_test[i]]))
#
# Testing Classifier Accuracy
sf = Scoring_Functions(y_pred=y_pred, y_true=y_test)
print(sf.scoring_results())