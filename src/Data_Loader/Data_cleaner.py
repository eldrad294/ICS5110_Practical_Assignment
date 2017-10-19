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
