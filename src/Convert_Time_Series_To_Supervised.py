import seaborn as sns
from pandas import DataFrame, concat
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
path = '../data/EEG/EEGEyeState.csv'
df = pd.read_csv(path)
df_headers = df.columns
featureCount = 14
#
# N-Step Univariate Forecasting Shift
shifted_df = series_to_supervised(data=df, n_in=1, n_out=1, dropnan=True)
shifted_df_headers = shifted_df.columns
shifted_feature_count = len(shifted_df_headers)
#
# Plot Pearson's Correlation Matrix for shifted_df
print(shifted_df_headers)
eegDF = pd.DataFrame(data=shifted_df[0:shifted_feature_count], columns=shifted_df_headers[0:shifted_feature_count])
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
#
plt.show()
#
# Store the upper triangle of the correlation matrix into an Excel sheet
corrUpperTri = corr.where(mask)
writer = pd.ExcelWriter('../data/EEG/EEG_Shifted_Correlation_Matrix.xlsx')
corrUpperTri.to_excel(writer, 'CorrelationMatrix')
writer.save()
#
# Plot Box Plots
def plotBoxPlots(data, arrLabels, titleSuffix):
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(30)

    ax1 = fig.add_subplot(2, 1, 1)
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
    #
    ax2 = fig.add_subplot(2, 1, 2)
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
# Stripping off Y column ('eyeDetection') from dataframe
shifted_df_wo_Y = shifted_df.drop('var15(t)', 1)
shifted_df_wo_Y = shifted_df_wo_Y.drop('var15(t-1)', 1)
shifted_df_wo_Y_count = len(shifted_df_wo_Y.columns)
outLierIndexes = set()
upperLimit = 5000
for x in range(shifted_df_wo_Y_count):
    shifted_df_wo_Y = np.array(shifted_df_wo_Y)
    outLiers = np.where(shifted_df_wo_Y[:, x] > upperLimit)[0]
    if len(outLiers) > 0:
        [(outLierIndexes.add(outLiers[i])) for i in range(len(outLiers))]
#
outLierIndexes = list(outLierIndexes)
outLierIndexes.sort()
print('Extreme outliers\nTotal:   ', len(outLierIndexes), \
    '\nIndexes: ', outLierIndexes)
[(print(DataFrame(shifted_df_wo_Y).iloc[[outlier]])) for outlier in outLierIndexes]
#
eegDataNoOutLiers = np.delete(shifted_df_wo_Y, outLierIndexes, 0)
plotBoxPlots(eegDataNoOutLiers[:, 0:shifted_df_wo_Y_count],
             shifted_df[0:shifted_df_wo_Y_count],
             'Extreme outliers excluded')
