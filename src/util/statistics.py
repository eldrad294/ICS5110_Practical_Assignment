import scipy.stats as st
import numpy as np
#
def get_confidence_interval(df, x , confidence=.95):
    """ Calculates the confidence interval of the passed dataset """
    min_confidence_interval, max_confidence_interval = st.norm.interval(confidence, loc=np.mean(df[x]), scale=st.sem(df[x]))
    return min_confidence_interval, max_confidence_interval