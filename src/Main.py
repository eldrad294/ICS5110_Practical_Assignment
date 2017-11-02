#
# Import Libraries and Dependencies
from src.Data_Loader.Load_data import Data_Loader
from src.Machine_Learning.Naive_Bayes import Naive_Bayes
from src.Machine_Learning.Support_Vector_Machine import SVM
import src.Graph_Handles.Graph_Handler as GH
#
allowed_features = ['Passengerid','Survived','Pclass','Sex','Age','Sibsp','Parch','Ticket', 'Fare', 'Embarked']
#
# Enter 2 Features, and data_mode [Train/Test]
while True:
    feature_1 = input("Enter first feature: ").lower().title()
    feature_2 = input("Enter second feature: ").lower().title()
    feature_3 = input("Enter third feature: ").lower().title()
    graphic_mode = input("Enter graphic mode: 1[Enabled], 2[Disabled]")
    #
    try:
        graphic_mode = int(graphic_mode)
        if feature_1 in allowed_features and feature_2 in allowed_features and feature_3 in allowed_features and \
                        graphic_mode in [1,2]:
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
                  '\n* Embarked\n')
            print('Graphic option must be 1[Enabled] or 2[Disabled]')
    except Exception as e:
        print('Graphic option must be 1[Enabled] or 2[Disabled]')
#
# Load the training data frame
data_loader_obj = Data_Loader(feature_1=feature_1,
                              feature_2=feature_2,
                              feature_3=feature_3,
                              data_mode="train")
df = data_loader_obj.get_data()
#
# Load the testing data frame
data_loader_obj = Data_Loader(feature_1=feature_1,
                              feature_2=feature_2,
                              feature_3=feature_3,
                              data_mode="test")
dft = data_loader_obj.get_data()
#
# Plot training set as scatter plot graph
# GH.plot_graph(features=df,
#               feature_1_name=feature_1,
#               feature_2_name=feature_2)
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
                validation_features.append([row[0],row[1], row[2]])
                validation_labels.append(row[3])
            else:
                train_features.append([row[0],row[1], row[2]])
                train_labels.append(row[3])
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
        #
        # Plot graph of estimated values
        # GH.scatter_plot_generator(graphic_mode=graphic_mode,
        #                           features=validation_features,
        #                           feature_1_name=feature_1,
        #                           feature_2_name=feature_2,
        #                           labels=predicted)
        GH.scatter_plot_generator_3D(graphic_mode=graphic_mode,
                                  features=validation_features,
                                  feature_1_name=feature_1,
                                  feature_2_name=feature_2,
                                  feature_3_name=feature_3,
                                  labels=predicted)
        #
        predicted_scores.append(accuracy_score)
        print('K_Fold sample accuracy: [' + str(accuracy_score) + ']')
        counter += 10
    #
    # Calculates the mean predicted score based off all the tests
    overall_score = sum(predicted_scores)/len(predicted_scores)
    print('Naive Bayes mean score of: ['+str(overall_score)+']')
    #
    # We use the testing set of data on the classifier that has been trained
    predicted = nb.predict(feature_array=dft)
    #
    # Plot graph of estimated values
    # GH.scatter_plot_generator(graphic_mode=1,
    #                           features=dft,
    #                           feature_1_name=feature_1,
    #                           feature_2_name=feature_2,
    #                           labels=predicted)
    GH.scatter_plot_generator_3D(graphic_mode=1,
                                 features=dft,
                                 feature_1_name=feature_1,
                                 feature_2_name=feature_2,
                                 feature_3_name=feature_3,
                                 labels=predicted)
    #
elif MLM == '2':
    #
    # Request SVM inputs
    supported_kernels = ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']
    #
    while True:
        #
        kernel = input("Enter kernel for SVM calculation. Kernel must be one of the following:"
                       "\nlinear"
                       "\nrbf"
                       "\npoly"
                       "\nsigmoid"
                       "\nprecomputed\n").lower()
        #
        if kernel in supported_kernels:
            break
        else:
            print('Incorrect Input. Try again.')
        #
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
                validation_features.append([row[0], row[1], row[2]])
                validation_labels.append(row[3])
            else:
                train_features.append([row[0], row[1], row[2]])
                train_labels.append(row[3])
            i += 1
        #
        svm = SVM(features_matrix=train_features,
                  labels=train_labels,
                  kernel=kernel,
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
        #
        # Plot graph of estimated values
        # GH.scatter_plot_generator(graphic_mode=graphic_mode,
        #                           features=validation_features,
        #                           feature_1_name=feature_1,
        #                           feature_2_name=feature_2,
        #                           labels=predicted)
        GH.scatter_plot_generator_3D(graphic_mode=graphic_mode,
                                     features=validation_features,
                                     feature_1_name=feature_1,
                                     feature_2_name=feature_2,
                                     feature_3_name=feature_3,
                                     labels=predicted)
        #
        predicted_scores.append(accuracy_score)
        print('K_Fold sample accuracy: [' + str(accuracy_score) + ']')
        counter += 10
    #
    # Calculates the mean predicted score based off all the tests
    overall_score = sum(predicted_scores) / len(predicted_scores)
    print('SVM mean score of: [' + str(overall_score) + ']')
    #
    #
    # We use the testing set of data on the classifier that has been trained
    predicted = svm.predict(feature_array=dft)
    #
    # Plot graph of estimated values
    # GH.scatter_plot_generator(graphic_mode=1,
    #                           features=dft,
    #                           feature_1_name=feature_1,
    #                           feature_2_name=feature_2,
    #                           labels=predicted)
    GH.scatter_plot_generator_3D(graphic_mode=1,
                                 features=dft,
                                 feature_1_name=feature_1,
                                 feature_2_name=feature_2,
                                 feature_3_name=feature_3,
                                 labels=predicted)
#
elif MLM == '3':
    pass
elif MLM == '4':
    pass
#

