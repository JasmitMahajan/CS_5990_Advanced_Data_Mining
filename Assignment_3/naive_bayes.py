#-------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: naive_bayes.py
# SPECIFICATION: This program will read the weather_training.csv file and classify each test instance from the file weather_test. Print
# the naive bayes accuracy calculated after all of the predictions.
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hour and 15 min
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 10)
attributes = ['Year', 'Month', 'Day', 'Hour', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
            'Visibility (km)', 'Pressure (millibars)', 'Temperature (C)']

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#reading the training data
fd_train = pd.read_csv('weather_training.csv')
fd_train['Formatted Date'] = pd.to_datetime(pd.read_csv('weather_training.csv')['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
fd_train['Year'] = fd_train['Formatted Date'].apply(lambda x: x.year)
fd_train['Month'] = fd_train['Formatted Date'].apply(lambda x: x.month)
fd_train['Day'] = fd_train['Formatted Date'].apply(lambda x: x.day)
fd_train['Hour'] = fd_train['Formatted Date'].apply(lambda x: x.hour)
fd_train['Humidity'] = fd_train['Humidity'].apply(lambda x: x*100)
fd_train = fd_train.astype({'Humidity': 'int', 'Wind Speed (km/h)': 'int', 'Visibility (km)': 'int', 'Pressure (millibars)': 'int', 'Temperature (C)': 'int'})
fd_train = fd_train.drop('Formatted Date', axis=1)

#reading the test data
fd_test = pd.read_csv('weather_test.csv')
fd_test['Formatted Date'] = pd.to_datetime(pd.read_csv('weather_test.csv')['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
fd_test['Year'] = fd_test['Formatted Date'].apply(lambda x: x.year)
fd_test['Month'] = fd_test['Formatted Date'].apply(lambda x: x.month)
fd_test['Day'] = fd_test['Formatted Date'].apply(lambda x: x.day)
fd_test['Hour'] = fd_test['Formatted Date'].apply(lambda x: x.hour)
fd_test['Humidity'] = fd_test['Humidity'].apply(lambda x: x*100)

fd_test = fd_test.astype({'Humidity': 'int', 'Wind Speed (km/h)': 'int', 'Visibility (km)': 'int', 'Pressure (millibars)': 'int', 'Temperature (C)': 'int'})
fd_test = fd_test.drop('Formatted Date', axis=1)

X_training = fd_train[attributes[:len(attributes)-1]]
Y_training = fd_train[attributes[len(attributes)-1]]

# Use test set instead of test split
X_test = fd_test[attributes[:len(attributes)-1]]
Y_test = fd_test[attributes[len(attributes)-1]]

#update the training class values according to the discretization (11 values only)
#update the test class values according to the discretization (11 values only)

disc = KBinsDiscretizer(n_bins=11, encode='ordinal').fit(X_training)
X_training = disc.transform(X_training)

disc = KBinsDiscretizer(n_bins=11, encode='ordinal').fit(X_test)
X_test = disc.transform(X_test)

# #fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, Y_training)
#
# #make the naive_bayes prediction for each test sample and start computing its accuracy
# #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
# #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
# #print the naive_bayes accuracy
Y_prediction = clf.predict(X_test)

accurate_prediction = 0
for i in range(len(Y_prediction)):
    difference = 100 * (abs(Y_prediction[i] - Y_test[i]) / Y_test[i])
    if 15 > difference > -15:
        accurate_prediction += 1

accuracy = accurate_prediction / len(Y_prediction)

print(f'naive_bayes accuracy: {accuracy}')



