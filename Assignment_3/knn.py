#-------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: knn.py
# SPECIFICATION: This program will read the weather_training.csv and estimate the temperature value for
# each data point in the weather_test.csv. Create a grid search trying multiple values for KNN hyperparameters
# and update and print the highest accuracy.
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#importing some Python libraries
#importing some Python libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import KBinsDiscretizer

#defining the hyperparameter values of KNN
k_vals = [i for i in range(1, 20)]
p_vals = [1, 2]
w_vals = ['uniform', 'distance']

features = ['Year', 'Month', 'Day', 'Hour', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
            'Visibility (km)', 'Pressure (millibars)', 'Temperature (C)']

#reading the test data
#hint: to convert values to float while reading them -> np.array(df_test.values)[:,-1].astype('f')
fd_test = pd.read_csv('weather_test.csv')
fd_test['Formatted Date'] = pd.to_datetime(pd.read_csv('weather_test.csv')['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
fd_test['Year'] = fd_test['Formatted Date'].apply(lambda x: x.year)
fd_test['Month'] = fd_test['Formatted Date'].apply(lambda x: x.month)
fd_test['Day'] = fd_test['Formatted Date'].apply(lambda x: x.day)
fd_test['Hour'] = fd_test['Formatted Date'].apply(lambda x: x.hour)
fd_test = fd_test.astype({'Wind Bearing (degrees)': 'float', 'Year': 'float', 'Month': 'float', 'Day': 'float', 'Hour': 'float'})
fd_test = fd_test.drop('Formatted Date', axis=1)

#reading the training data
fd_train = pd.read_csv('weather_training.csv')
fd_train['Formatted Date'] = pd.to_datetime(pd.read_csv('weather_training.csv')['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
fd_train['Year'] = fd_train['Formatted Date'].apply(lambda x: x.year)
fd_train['Month'] = fd_train['Formatted Date'].apply(lambda x: x.month)
fd_train['Day'] = fd_train['Formatted Date'].apply(lambda x: x.day)
fd_train['Hour'] = fd_train['Formatted Date'].apply(lambda x: x.hour)
fd_train = fd_train.astype({'Wind Bearing (degrees)': 'float', 'Year': 'float', 'Month': 'float', 'Day': 'float', 'Hour': 'float'})
fd_train = fd_train.drop('Formatted Date', axis=1)

accuracy_max = 0
#loop over the hyperparameter values (k, p, and w) ok KNN
for k in k_vals:
    for p in p_vals:
        for w in w_vals:
            #fitting the knn to the data
            X_training = fd_train[features[:len(features)-1]]
            Y_training = fd_train[features[len(features)-1]]

            # Use test set instead of test split
            X_test = fd_test[features[:len(features)-1]]
            y_test = fd_test[features[len(features)-1]]

            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, Y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))

            #--> add your Python code here
            y_prediction = clf.predict(X_test)
            accurate_prediction = 0
            for i in range(len(y_prediction)):
                difference = 100 * abs((y_prediction[i] - y_test[i]) / y_test[i])
                if 15 > difference > -15:
                    accurate_prediction += 1

            accuracy = accurate_prediction / len(y_prediction)

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"

            #--> add your Python code here
            accuracy_max = max(accuracy_max, accuracy)
            print(f'Highest KNN accuracy so far: {accuracy_max}, Parameters: k={k}, p={p}, w={w}')

