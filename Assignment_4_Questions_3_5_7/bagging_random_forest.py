#-------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: bagging_random_forest.py
# SPECIFICATION: Read the optdigits.tra and optdigits.names
# to see it was transformed to speed-up the learning process
# Building a base classifier by using a single decision tree
# and a Random Forest classifier to recognize the digits.
# FOR: CS 5990- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
with open('optdigits.tra', 'r') as trainingFile:
   reader = csv.reader(trainingFile)
   for i, row in enumerate(reader):
      dbTraining.append(row)

#reading the test data from a csv file and populate dbTest
with open('optdigits.tes', 'r') as testFile:
   reader = csv.reader(testFile)
   for i, row in enumerate(reader):
      dbTest.append(row)

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
      classVotes.append([0,0,0,0,0,0,0,0,0,0])

print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  for value in bootstrapSample:
      X_training.append(value[:len(value)-1])
      y_training.append(value[len(value)-1])

  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)

  trueLabel = 0
  for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      prediction = int(clf.predict([testSample[:len(testSample)-1]])[0])
      classVotes[i][prediction] += 1

      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         trueLabel += 1 * (prediction == int(testSample[len(testSample)-1]))

  if k == 0: #for only the first base classifier, print its accuracy here
     accuracy = trueLabel / len(dbTest)
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(accuracy))
     print("")

  #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
  trueLabel = 0
  for i, testSample in enumerate(dbTest):
      majorityVote = int(classVotes[i].index(max(classVotes[i])))
      trueLabel += 1 * (majorityVote == int(testSample[len(testSample)-1]))

#printing the ensemble accuracy here
accuracy = trueLabel / len(dbTest)
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

truthLabel = 0
#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
for i, testSample in enumerate(dbTest):
   prediction_rf = int(clf.predict([testSample[:len(testSample)-1]])[0])

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
   trueLabel += 1 * (prediction_rf == int(testSample[len(testSample)-1]))

#printing Random Forest accuracy here
accuracy = trueLabel/len(dbTest)
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
