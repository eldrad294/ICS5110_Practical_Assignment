import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
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
def scatter_plot_generator_3D(graphic_mode, features, feature_1_name, feature_2_name, feature_3_name, labels):
    """ Takes 3 feature lists, and attempts to plot them on 3D """
    #
    if graphic_mode != 1:
        return
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, row in enumerate(features):
        x, y, z = row
        if labels[i] == 0:
            ax.scatter(x, y, z, c='r', marker='o') #Fatality
        elif labels[i] == 1:
            ax.scatter(x, y, z, c='g', marker='o') #Survived
    #
    ax.set_xlabel(feature_1_name)
    ax.set_ylabel(feature_2_name)
    ax.set_zlabel(feature_3_name)
    #
    plt.title(feature_1_name + ' vs ' + feature_2_name + ' vs ' + feature_3_name)
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