import numpy as np
import matplotlib.pyplot as plt
import src.util.statistics as st
import seaborn as sns
import pandas as pd
#
class GraphFactory():
    #
    def __init__(self,save_path):
        self.save_path = save_path
    #
    def scatter_plot(self,f1,f2,Y,display, confidence_interval=None):
        """ Plots a scatter plot based on 2 input columns.
        Points marked in blue signals eye closed, red signals eye open """
        #
        x1, x2, Y = np.asarray(f1), np.asarray(f2), np.asarray(Y)
        x1_confidence_interval_min, x1_confidence_interval_max = st.get_confidence_interval(f1, f1.columns.values, confidence_interval)
        x2_confidence_interval_min, x2_confidence_interval_max = st.get_confidence_interval(f2, f2.columns.values, confidence_interval)
        #
        for i in range(len(Y)):
            if confidence_interval is not None:
                if x1[i] >= x1_confidence_interval_min and x1[i] <= x1_confidence_interval_max \
                        and x2[i] >= x2_confidence_interval_min and x2[i] <= x2_confidence_interval_max:
                    plt.scatter(x1[i], x2[i], c='b' if Y[i] == 0 else 'r')
            else:
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
    def correlation_matrix(self, df, display, confidence_interval=None):
        """ Plots Correlation Matrix """
        #
        # Prune dataframe if confidence_interval is declared
        if confidence_interval is not None:
            for column in df:
                min_confidence_interval, max_confidence_interval = st.get_confidence_interval(df[[column]],column,confidence_interval)
                df_c = []
                for value in df[column]:
                    if value >= min_confidence_interval and value <= max_confidence_interval:
                        df_c.append(value)
                df[column] = pd.DataFrame({column: df_c})
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