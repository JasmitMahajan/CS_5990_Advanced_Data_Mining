# -------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: decision_tree.py
# SPECIFICATION: This program will read the training files and build a decision tree. You would repeat this for 10 times
# and would average the accuracies as the final classification performance of each model.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 2 hrs for this code
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)  # reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:, 1:]  # creating a training matrix without the id (NumPy library)

    # transform the original training features to numbers and add them to the 5D array X.
    # For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]].
    # The feature Marital Status must be one-hot-encoded and Taxable Income must be converted to a float.

    marital_status_list = ['Divorced', 'Married', 'Single']
    taxable_income_max = 0
    taxable_income_min = 10000

    for i in range(len(df)):
        attribute_list = list(df.values[i])
        instance_list = []

        refund_status = attribute_list[1]
        if 'Yes' in attribute_list[1]:
            instance_list.append(1)
        else:
            instance_list.append(0)

        marital_status = attribute_list[2]
        marital_status_list_initializer = [0, 0, 0]

        for j in range(len(marital_status_list)):
            if attribute_list[2] in marital_status_list[j]:
                instance_list.append(1)
            else:
                instance_list.append(0)

        taxable_income = int(attribute_list[3].strip('k'))
        instance_list.append(taxable_income)
        taxable_income_min = min(taxable_income, taxable_income_min)
        taxable_income_max = max(taxable_income, taxable_income_max)

        X.append(instance_list)

        # transform the original training classes to numbers and add them to the vector Y.
        # For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        if 'Yes' in attribute_list[4]:
            Y.append(1)
        else:
            Y.append(0)

    for j in range(len(df)):
        taxable_income = X[j][4]
        normalized_income = (taxable_income - taxable_income_min) / (taxable_income_max - taxable_income_min)
        X[j][4] = normalized_income


    accuracy_list = []
    # loop your training and test tasks 10 times here
    for i in range(10):
        correct_vals = 0
        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()
        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        dTest = pd.read_csv('cheat_test.csv', sep=',', header=0)
        data_test = dTest.values

        X_test = []
        Y_test = []

        taxable_income_max_test = 0
        taxable_income_min_test = 10000

        for data in data_test:
        # transform the features of the test instances to numbers following the same strategy done during training,
        #  and then use the decision tree to make the class prediction. For instance:
        # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        # --> add your Python code here
            attribute_list = list(data)
            instance_list = []

            refund_status = attribute_list[1]
            if 'Yes' in attribute_list[1]:
                instance_list.append(1)
            else:
                instance_list.append(0)

            marital_status = attribute_list[2]
            marital_status_list_initializer = [0, 0, 0]

            for j in range(len(marital_status_list)):
                if attribute_list[2] in marital_status_list[j]:
                    instance_list.append(1)
                else:
                    instance_list.append(0)

            taxable_income = int(attribute_list[3].strip('k'))
            instance_list.append(taxable_income)
            taxable_income_min_test = min(taxable_income, taxable_income_min_test)
            taxable_income_max_test = max(taxable_income, taxable_income_max_test)

            X_test.append(instance_list)

            # transform the original training classes to numbers and add them to the vector Y.
            # For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
            if 'Yes' in attribute_list[4]:
                Y_test.append(1)
            else:
                Y_test.append(0)

        for j in range(len(data_test)):
            taxable_income = X_test[j][4]
            normalized_income = (taxable_income - taxable_income_min_test) / (taxable_income_max_test - taxable_income_min_test)
            X_test[j][4] = normalized_income

        # compare the prediction with the true label of the test instance to start calculating the model accuracy.
        class_predicted = clf.predict(X_test)

        temp = 0
        for data in data_test:
            if data[4] in 'Yes' and class_predicted[temp] == 1:
                correct_vals += 1
            elif data[4] in 'No' and class_predicted[temp] == 0:
                correct_vals += 1
            temp += 1

        accuracy_list.append(correct_vals / len(data_test))

    # find the average accuracy of this model during the 10 runs (training and test set)
    total_sum = 0
    for value in accuracy_list:
        total_sum += value
    average_accuracy = total_sum / len(accuracy_list)

    # print the accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    print(f'final accuracy when training on {ds}: {average_accuracy}')