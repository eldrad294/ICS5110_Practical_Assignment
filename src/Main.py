#
# Import Libraries and Dependencies
from src.Data_Loader.Load_data import Data_Loader
from src.Machine_Learning.Naive_Bayes import Naive_Bayes
from src.Machine_Learning.Support_Vector_Machine import SVM
#
allowed_features = ['Passengerid','Survived','Pclass','Sex','Age','Sibsp','Parch','Ticket', 'Fare', 'Cabin', 'Embarked']
#
# Enter 2 Features, and data_mode [Train/Test]
while True:
    feature_1 = input("Enter first feature: ").lower().title()
    feature_2 = input("Enter second feature: ").lower().title()
    #
    if feature_1 in allowed_features and feature_2 in allowed_features:
        break
    else:
        print('Features must be of the following:'
              '\n* Passengerid'
              '\n* Survived'
              '\n* Pclass'
              '\n* Sex'
              '\n* Age'
              '\n* Sibsp'
              '\n* Parch'
              '\n* Ticket'
              '\n* Fare'
              '\n* Cabin'
              '\n* Embarked\n')
#
data_loader_obj = Data_Loader(feature_1=feature_1,
                              feature_2=feature_2,
                              data_mode="train")
df = data_loader_obj.get_data()
#
# Enter Machine Learning Method
while True:
    MLM = input('Enter which machine learning method to utilise as follows:'
                '\n1..Naive Bayes Classification'
                '\n2..Support Vector Machine Classification')
    try:
        if int(MLM) >= 0 and int(MLM) < 5:
            break
        else:
            print('Must enter a numeric digit between 1 and 5.')
    except Exception:
        print('Must enter a numeric digit between 1 and 5.')
#
k_fold = 10
counter = 10
predicted_scores = []
#
if MLM == '1':
    while counter < len(df):
        #
        # Split original dataframe into the training, and testing sample
        validation_features = []
        validation_labels = []
        train_features = []
        train_labels = []
        #
        # Populate test_df and train_df respectively
        i = 1
        for row in df:
            if i >= counter-k_fold and i < counter:
                validation_features.append([row[0],row[1]])
                validation_labels.append(row[2])
            else:
                train_features.append([row[0],row[1]])
                train_labels.append(row[2])
            i += 1
        #
        nb = Naive_Bayes(features_matrix=train_features,
                         labels=train_labels)
        #
        # Trains the Naive Bayes classifier using the data we passed
        nb.fit_training_set()
        #
        # We use the validation set of data to test our classifier
        predicted = nb.predict(feature_array=validation_features)
        #
        # We calculate the accuracy score of the sample run
        accuracy_score = nb.calculate_accuracy_score(labels=validation_labels,
                                                     predicted=predicted)
        predicted_scores.append(accuracy_score)
        print('K_Fold sample accuracy: [' + str(accuracy_score) + ']')
        counter += 10
    #
    # Calculates the mean predicted score based off all the tests
    overall_score = sum(predicted_scores)/len(predicted_scores)
    print('Naive Bayes mean score of: ['+str(overall_score)+']')
elif MLM == '2':
    while counter < len(df):
        #
        # Split original dataframe into the training, and testing sample
        validation_features = []
        validation_labels = []
        train_features = []
        train_labels = []
        #
        # Populate test_df and train_df respectively
        i = 1
        for row in df:
            if i >= counter - k_fold and i < counter:
                validation_features.append([row[0], row[1]])
                validation_labels.append(row[2])
            else:
                train_features.append([row[0], row[1]])
                train_labels.append(row[2])
            i += 1
        #
        svm = SVM(features_matrix=train_features,
                  labels=train_labels,
                  kernel="rbf",
                  gamma=0.09,
                  C=5)
        #
        # Trains the Naive Bayes classifier using the data we passed
        svm.fit_training_set()
        #
        # We use the validation set of data to test our classifier
        predicted = svm.predict(feature_array=validation_features)
        #
        # We calculate the accuracy score of the sample run
        accuracy_score = svm.calculate_accuracy_score(labels=validation_labels,
                                                      predicted=predicted)
        predicted_scores.append(accuracy_score)
        print('K_Fold sample accuracy: [' + str(accuracy_score) + ']')
        counter += 10
    #
    # Calculates the mean predicted score based off all the tests
    overall_score = sum(predicted_scores) / len(predicted_scores)
    print('SVM mean score of: [' + str(overall_score) + ']')
elif MLM == '3':
    pass
elif MLM == '4':
    pass
