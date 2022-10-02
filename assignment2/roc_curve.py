# -------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: roc_curve.py
# SPECIFICATION: This program will read the cheat_data.csv file, split the training and test data, and compute the ROC
# curve for a decision tree classifier.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 30 min for this code
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
df = pd.read_csv('cheat_data.csv', sep=',', header=0)
data_training = np.array(df.values)[:, 1:]

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
X = []

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
Y = []

marital_status_list = ['Divorced', 'Married', 'Single']
taxable_income_max = 0
taxable_income_min = 10000

for i in range(len(df)):
    attribute_list = list(df.values[i])
    instance_list = []

    refund_status = attribute_list[0]
    if 'Yes' in attribute_list[0]:
        instance_list.append(1)
    else:
        instance_list.append(0)

    marital_status = attribute_list[1]
    marital_status_list_initializer = [0, 0, 0]

    for j in range(len(marital_status_list)):
        if attribute_list[1] in marital_status_list[j]:
            instance_list.append(1)
        else:
            instance_list.append(0)

    taxable_income = int(attribute_list[2].strip('k'))
    instance_list.append(taxable_income)
    taxable_income_min = min(taxable_income, taxable_income_min)
    taxable_income_max = max(taxable_income, taxable_income_max)

    X.append(instance_list)

        # transform the original training classes to numbers and add them to the vector Y.
        # For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    if 'Yes' in attribute_list[3]:
        Y.append(1)
    else:
        Y.append(0)

for j in range(len(df)):
    taxable_income = X[j][4]
    normalized_income = (taxable_income - taxable_income_min) / (taxable_income_max - taxable_income_min)
    X[j][4] = normalized_income

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainY, testy = train_test_split(X, Y, test_size = 0.30)

# generate a no skill prediction (random classifier - scores should be all zero)
# --> add your Python code here
ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainY)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifeirs (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()