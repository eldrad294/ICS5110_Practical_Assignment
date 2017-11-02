import numpy as np
#
def clean_data_frame(df):
    """ Converts the data in proper, readable format. """
    #
    # Convert dataframe to numpy array
    df = np.array(df)
    #
    # Remove missing elements from the dataframe
    df = remove_missing_data(df=df)
    #
    # Check for sex feature, and normalize it
    df = check_gender_and_convert(df=df)
    #
    # Check for embarked feature, and convert it
    df = check_embarked_and_convert(df=df)
    #
    return df
#
def remove_missing_data(df):
    """ Removes rows which have missing data in the the respective columns """
    clean_df = []
    for row in df:
        try:
            if np.isnan(row[0]):
                continue
        except Exception:
            if str(row[0]) is None or str(row[0]) == "":
                continue
        #
        try:
            if np.isnan(row[1]):
                continue
        except Exception:
            if str(row[1]) is None or str(row[1]) == "":
                continue
        #
        try:
            if np.isnan(row[2]):
                continue
        except Exception:
            if str(row[2]) is None or str(row[2]) == "":
                continue
        #
        clean_df.append(row)
    return clean_df
#
def check_gender_and_convert(df):
    """ Makes a check for the feature 'Sex', and if found, normalizes it to 0,1 """
    gender = ['male', 'female']
    if df[0][0] in gender:
        for row in df:
            row[0] = normalize_sex(row[0])
    #
    if df[0][1] in gender:
        for row in df:
            row[1] = normalize_sex(row[1])
    #
    if df[0][2] in gender:
        for row in df:
            row[2] = normalize_sex(row[2])
    #
    return df
#
def check_embarked_and_convert(df):
    """ Makes a check for the feature 'Embarked', and if found, normalizes it to 0,1,2 """
    embarked = ['C','S','Q']
    if df[0][0] in embarked:
        for row in df:
            row[0] = convert_embarked(row[0].upper())
    #
    if df[0][1] in embarked:
        for row in df:
            row[1] = convert_embarked(row[1].upper())
    #
    if df[0][2] in embarked:
        for row in df:
            row[2] = convert_embarked(row[2].upper())
    #
    return df
#
def normalize_sex(sex):
    """ Returns 0 if male, 1 if female """
    if sex == "male":
        return 0
    elif sex == "female":
        return 1
    else:
        print('Invalid Data Mode.')
        exit(0)
#
def convert_embarked(embarked):
    """ Converts C=0, S=1, Q=2 """
    if embarked == "C":
        return 0
    elif embarked == "S":
        return 1
    elif embarked == "Q":
        return 2
    else:
        print('Invalid Data Mode.')
        exit(0)
