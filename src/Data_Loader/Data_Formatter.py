import pandas as pd
import src.statistics.statistics as st
from sklearn import preprocessing
#
class Data_Formatter:
    #
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.stats = st.Statistics()
        self.excluded = ('eyeDetection')
    #
    def get_df(self):
        """ Returns data frame """
        return self.df
    #
    def get_df_header(self):
        """ Returns data frame header """
        return self.df.columns
    #
    def get_confidence_df(self, confidence_interval):
        """ Returns data frame with excluded outliers """
        if confidence_interval is not None:
            for column in self.df:
                if column not in self.excluded:
                    min_confidence_interval, max_confidence_interval = self.stats.get_confidence_interval(self.df[[column]],
                                                                                                          column,
                                                                                                          confidence_interval)
                    df_c = []
                    # print(column)
                    # print('Min: ' + str(min_confidence_interval))
                    # print('Max: ' + str(max_confidence_interval))
                    # print('--------------')
                    for value in self.df[column]:
                        if value >= min_confidence_interval and value <= max_confidence_interval:
                            df_c.append(value)
                        else:
                            # We assign outliers a value of 9999
                            df_c.append(9999)
                    self.df[column] = pd.DataFrame({column: df_c})
        return self.df
    #
    def normalize_data(self):
        """ Normalizes dataset by estimating z-score """
        for column in self.df:
            if column not in self.excluded:
                self.df[column] = (self.df[column] - self.df[column].mean())/self.df[column].std()
        return self.df