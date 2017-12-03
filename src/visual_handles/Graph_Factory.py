import numpy as np
import matplotlib.pyplot as plt
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
        #
        for i in range(10000):
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
