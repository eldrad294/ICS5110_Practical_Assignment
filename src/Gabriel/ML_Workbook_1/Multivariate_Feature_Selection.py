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
# Cross-validating of dataset: Splitting dataset into sub samples for training and testing purposes
from sklearn.model_selection import train_test_split
df_pruned_shifted_Y = df_pruned_shifted['var15(t)']
df_pruned_shifted_X = df_pruned_shifted.drop('var15(t)', 1)
#
X_train, X_test, y_train, y_test = train_test_split(df_pruned_shifted_X, df_pruned_shifted_Y, test_size=0.2, random_state=0)
#
# Normalize shifted_df_X
from sklearn.preprocessing import normalize
X_train = normalize(X_train, norm='l2')
X_test = normalize(X_test, norm='l2')
#
# https://www.fabienplisson.com/choosing-right-features/
print('Random Forest Classifier')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
feature_importances = pd.DataFrame({'feature': df_pruned_shifted_X.columns, 'importance': np.round(rfc.feature_importances_, 6)})
feature_importances = feature_importances.sort_values('importance', ascending=False).set_index('feature')
#
print(feature_importances)
#
# Plot the feature importance of the forest
outward_path = 'D:\\Projects\\ICS5110_Practical_Assignment\\src\\Gabriel\\ML_Workbook_1\\Findings\\Plots_Multivariate\\Lag' + str(lag) + '\\'
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
from numpy import sort
from sklearn.feature_selection import SelectFromModel
#
model = RandomForestClassifier()
model.fit(X_train, y_train)
#
# make predictions for test data and evaluate
pred_y = model.predict(X_test)
#
# Testing Classifier Accuracy
from src.statistics.scoring_functions import Scoring_Functions
sf = Scoring_Functions(y_pred=pred_y, y_true=y_test)
print("RFC Accuracy: ")
print(sf.scoring_results())
print('-------------------------')
#
# fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # selecting features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_train_x = selection.transform(X_train)
    #
    # training model
    selection_model = RandomForestClassifier()
    selection_model.fit(select_train_x, y_train)
    #
    # evaluating model
    select_test_x = selection.transform(X_test)
    pred_y = selection_model.predict(select_test_x)
    sf = Scoring_Functions(y_pred=pred_y, y_true=y_test)
    print('Threshold: ' + str(thresh))
    print('Feature Count: ' + str(select_train_x.shape[1]))
    print(sf.scoring_results())
    print('-------------------------')
#
print('Gradient Boosting Classifier')
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
feature_importances = pd.DataFrame({'feature': df_pruned_shifted_X.columns, 'importance': np.round(gbc.feature_importances_, 6)})
feature_importances = feature_importances.sort_values('importance', ascending=False).set_index('feature')
#
print(feature_importances)
#
# Plot the feature importance of the forest
outward_path = 'D:\\Projects\\ICS5110_Practical_Assignment\\src\\Gabriel\\ML_Workbook_1\\Findings\\Plots_Multivariate\\Lag' + str(lag) + '\\'
import matplotlib.pyplot as plt
fig = plt.figure()
plt.title("Feature importance")
objects = list(feature_importances.axes[0])
y_pos = np.arange(len(objects))
feature_importance = np.array(feature_importances)
plt.bar(y_pos, feature_importance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
fig.savefig(outward_path + 'GradientBoostingClassifier_FeatureImportance.png')
#
from numpy import sort
from sklearn.feature_selection import SelectFromModel
#
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
#
# make predictions for test data and evaluate
pred_y = model.predict(X_test)
#
# Testing Classifier Accuracy
from src.statistics.scoring_functions import Scoring_Functions
sf = Scoring_Functions(y_pred=pred_y, y_true=y_test)
print("GBC Accuracy: ")
print(sf.scoring_results())
print('-------------------------')
#
# fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # selecting features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_train_x = selection.transform(X_train)
    #
    # training model
    selection_model = GradientBoostingClassifier()
    selection_model.fit(select_train_x, y_train)
    #
    # evaluating model
    select_test_x = selection.transform(X_test)
    pred_y = selection_model.predict(select_test_x)
    sf = Scoring_Functions(y_pred=pred_y, y_true=y_test)
    print('Threshold: ' + str(thresh))
    print('Feature Count: ' + str(select_train_x.shape[1]))
    print(sf.scoring_results())
    print('-------------------------')