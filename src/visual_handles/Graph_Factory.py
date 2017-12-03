import numpy as np
import matplotlib.pyplot as plt
import src.util.statistics as st
#
class GraphFactory():
    #
    def __init__(self,save_path):
        self.save_path = save_path
    #
    def scatter_plot(self,f1,f2,Y,display):
        """ Plots a scatter plot based on 2 input columns.
        Points marked in blue signals eye closed, red signals eye open """
        #
        x1, x2, Y = np.asarray(f1), np.asarray(f2), np.asarray(Y)
        x1_confidence_interval_min, x1_confidence_interval_max = st.get_confidence_interval(f1, f1.columns.values, .99)
        x2_confidence_interval_min, x2_confidence_interval_max = st.get_confidence_interval(f2, f2.columns.values, .99)
        #
        for i in range(len(Y)):
            if x1[i] > x1_confidence_interval_min and x1[i] < x1_confidence_interval_max \
                    and x2[i] > x2_confidence_interval_min and x2[i] < x2_confidence_interval_max:
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
