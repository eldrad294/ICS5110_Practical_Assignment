import pandas as pd
#
class Data_Formatter:
    #
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    #
    def get_df(self):
        """ Returns data frame """
        return self.df