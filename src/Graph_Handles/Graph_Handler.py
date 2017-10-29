import matplotlib.pyplot as plt
#
def scatter_plot_generator(graphic_mode, features, feature_1_name, feature_2_name, labels):
    """ Takes 2 feature lists, and attempts to plot them on 2D """
    #
    if graphic_mode != 1:
        return
    #
    for i, feature in enumerate(features):
        if labels[i] == 0:
            plt.plot(feature[0], feature[1], 'ro') #Fatality
        elif labels[i] == 1:
            plt.plot(feature[0], feature[1], 'go') #Survived
    #
    plt.xlabel(feature_1_name)
    plt.ylabel(feature_2_name)
    plt.title(feature_1_name + ' vs ' + feature_2_name)
    plt.show()
#
def plot_graph(features, feature_1_name, feature_2_name):
    """ Plots a simple x, y graph """
    #
    for feature in features:
        plt.plot(feature[0], feature[1], 'bo')
    #
    plt.xlabel(feature_1_name)
    plt.ylabel(feature_2_name)
    plt.title(feature_1_name + ' vs ' + feature_2_name)
    plt.show()