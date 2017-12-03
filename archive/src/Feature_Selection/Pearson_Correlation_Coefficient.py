import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#
class PearsonCoefficient:
    #
    def __init__(self, df):
        sns.set(style="white")
        #
        # Compute the correlation matrix
        corr = df.corr()
        #
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        #
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(100, 100))
        #
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        #
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()
