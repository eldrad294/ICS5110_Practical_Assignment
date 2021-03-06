import pandas as pd
import src.Data_Loader.Data_cleaner as dc
from src.Feature_Selection.Pearson_Correlation_Coefficient import PearsonCoefficient
#
class Data_Loader:
    #
    def __init__(self, feature_1, feature_2, feature_3, data_mode="train"):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.feature_3 = feature_3
        self.data_mode = data_mode
        #
        if self.data_mode == "train" or self.data_mode is None:
            self.df = pd.read_csv('../data/train.csv')
        elif self.data_mode == "test":
            self.df = pd.read_csv('../data/test.csv')
        else:
            print('Invalid Data Mode.')
            exit(0)
    #
    def get_data(self):
        """ Retrieves data based on pre-selected features, cleans it, and calculates the k-fold """
        if self.data_mode == "train":
            df = dc.clean_data_frame(self.df[[self.feature_1, self.feature_2, self.feature_3, "Survived"]])
        elif self.data_mode is None:
            df = dc.PC_clean_data_frame(self.df)
        elif self.data_mode == "test":
            df = dc.clean_data_frame(self.df[[self.feature_1, self.feature_2, self.feature_3]])
        return df
    #
    def suggest_features(self, df):
        #
        # Calculate Correlation Matrix on features and suggest them to user
        print("Calculating Correlation Matrix")
        PearsonCoefficient(df=df)


