import numpy as np
import matplotlib.pyplot as plt
import src.statistics.statistics as st
import seaborn as sns
import pandas as pd
#
class GraphFactory():
    #
    def __init__(self,save_path):
        self.save_path = save_path
        self.stats = st.Statistics()
    #
    def scatter_plot(self,f1,f2,Y,display):
        """ Plots a scatter plot based on 2 input columns.
        Points marked in blue signals eye closed, red signals eye open """
        #
        x1, x2, Y = np.asarray(f1), np.asarray(f2), np.asarray(Y)
        #
        for i in range(len(Y)):
            if x1[i] != 9999 and x2[i] != 9999:
                plt.scatter(x1[i], x2[i], c='b' if Y[i] == 0 else 'r')
        #
        plt.title('Scatter plot ' + f1.columns.values + ' vs ' + f2.columns.values)
        plt.xlabel(f1.columns.values)
        plt.ylabel(f2.columns.values)
        #
        if display is True:
            plt.show()
        else:
            plt.savefig(self.save_path + 'Scatter plot ' + str(f1.columns.values) + ' vs ' + str(f2.columns.values) + '.png')
    #
    def correlation_matrix(self, df, display, confidence_interval):
        """ Plots Correlation Matrix """
        #
        f, ax = plt.subplots(figsize=(12, 12))
        corr = df.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax, vmin=-1, vmax=1)
        #
        if display is True:
            plt.show()
        else:
            plt.savefig(self.save_path + 'Correlation Matrix with ' + str(confidence_interval) + '% outliers excluded.png')