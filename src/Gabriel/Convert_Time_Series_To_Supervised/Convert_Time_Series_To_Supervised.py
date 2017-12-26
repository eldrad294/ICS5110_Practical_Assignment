import seaborn as sns
from pandas import DataFrame, concat, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
df_headers = df.columns
featureCount = 14
#
# N-Step Univariate Forecasting Shift
lag = 4
shifted_df = series_to_supervised(data=df, n_in=lag, n_out=1, dropnan=True)
shifted_df_headers = shifted_df.columns
shifted_feature_count = len(shifted_df_headers)
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
# Plot Pearson's Correlation Matrix for shifted_df
eegDF = pd.DataFrame(data=shifted_df_X[0:shifted_df_X_count], columns=shifted_columns[0:shifted_df_X_count])
#
# Compute the correlation matrix
corr = eegDF.corr()
#
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#
# Set up the matplotlib figure
fig = plt.figure()
fig.set_figheight(20)
fig.set_figwidth(20)
#
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Feature correlation matrix")
#
# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 20, as_cmap=True)
#
# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
#
# Store the upper triangle of the correlation matrix into an Excel sheet
corrUpperTri = corr.where(mask)
writer = pd.ExcelWriter('EEG_Shifted_Correlation_Matrix.xlsx')
corrUpperTri.to_excel(writer, 'CorrelationMatrix')
writer.save()
#
# Plot Pearson's Correlation Matrix for shifted_df without outliers
eegDF = pd.DataFrame(data=shifted_df_X[0:shifted_df_X_count], columns=shifted_columns[0:shifted_df_X_count])
outLierIndexes = get_outlier_indexes(eegDF, shifted_df_X_count, 5000)
eegDF = np.delete(eegDF, outLierIndexes, 0)
#
# Compute the correlation matrix
corr = eegDF.corr()
#
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#
# Set up the matplotlib figure
fig = plt.figure()
fig.set_figheight(20)
fig.set_figwidth(20)
#
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Feature correlation matrix excluding outliers")
#
# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 20, as_cmap=True)
#
# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
#
# Store the upper triangle of the correlation matrix into an Excel sheet
corrUpperTri = corr.where(mask)
writer = pd.ExcelWriter('EEG_Shifted_Correlation_Matrix_without_outliers.xlsx')
corrUpperTri.to_excel(writer, 'CorrelationMatrix')
writer.save()
#
# Plot Box Plots
def plotBoxPlots(data, arrLabels, titleSuffix):
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(30)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Feature range: full - ' + titleSuffix)
    suppress = ax1.boxplot(data
                           , sym='b.'
                           , vert=False
                           , whis='range'
                           , labels=arrLabels
                           , meanline=True
                           , showbox=True
                           , showfliers=True
                           )
    plt.show()
    #
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(30)
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Feature range: [5%, 95%] - ' + titleSuffix)
    suppress = ax2.boxplot(data
                           , sym='b.'
                           , vert=False
                           , whis=[5, 95]
                           , labels=arrLabels
                           , meanline=True
                           , showbox=True
                           , showfliers=False
                           )
    plt.show()
#
# Plotting Box Plots
outLierIndexes = get_outlier_indexes(shifted_df_X, shifted_df_X_count, 5000)
print('Extreme outliers\nTotal:   ', len(outLierIndexes), \
    '\nIndexes: ', outLierIndexes)
[(print(DataFrame(shifted_df_X).iloc[[outlier]])) for outlier in outLierIndexes]
#
eegDataNoOutLiers_X = np.delete(shifted_df_X, outLierIndexes, 0)
eegDataNoOutLiers_y = np.delete(shifted_df_y, outLierIndexes, 0)
plotBoxPlots(shifted_df_X[:, 0:shifted_df_X_count],
             shifted_columns,
             'Extreme outliers excluded')
shifted_df_X = eegDataNoOutLiers_X
shifted_df_y = eegDataNoOutLiers_y
#
# Autocorrelation Plot
# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
#
series = Series.from_csv(path, header=0)
print(series.values)
from statsmodels.graphics.tsaplots import plot_acf
if lag == 0:
    lag = None
plot_acf(series, lags=lag)
plt.show()
#
# Feature Selection using RandomForests
# https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestClassifier(n_estimators=500)
X_train = shifted_df_X
y_train = shifted_df_y
rf.fit(X_train, y_train)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = shifted_columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
print(feature_importances)
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importance")
objects = list(feature_importances.axes[0])
y_pos = np.arange(len(objects))
feature_importance = np.array(feature_importances)
plt.bar(y_pos, feature_importance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.show()
#
# Plot Scatter plots for Top N Features of Importance
