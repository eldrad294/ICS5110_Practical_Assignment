import pandas as pd
import src.Data_Loader.Data_cleaner as dc
#
class Data_Loader:
    #
    def __init__(self, feature_1, feature_2, data_mode="train"):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.data_mode = data_mode
        #
        if self.data_mode == "train":
            self.df = pd.read_csv('../data/train.csv')
        # elif self.data_mode == "test":
        #     self.df = pd.read_csv('../data/test.csv')
        else:
            print('Invalid Data Mode.')
            exit(0)
    #
    def get_data(self):
        """ Retrieves data based on pre-selected features, cleans it, and calculates the k-fold """
        if self.data_mode == "train":
            df = dc.clean_data_frame(self.df[[self.feature_1, self.feature_2, "Survived"]])
        elif self.data_mode == "test":
            df = dc.clean_data_frame(self.df[[self.feature_1, self.feature_2]])
        return df

