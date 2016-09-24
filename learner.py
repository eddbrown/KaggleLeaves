import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Load the training data
train_csv = pd.read_csv('./train.csv')

#Encode the species as a number (0-99)
encoder = LabelEncoder().fit(train_csv['species'])
y_train = encoder.transform(train_csv['species'])

#Prepare the training data
x_train = train_csv.drop(['id', 'species'], axis=1).values
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#Set the parameters on the SVM and then run the algorithm
clf = svm.SVC(gamma= 0.001, C=100, probability=True)
clf.fit(x_train,y_train)

#Load the test data
test = pd.read_csv('./test.csv')
test_ids = test.pop('id')

#Transform the test data in the same manner as the training data
x_test = test.values
x_test = scaler.transform(x_test)

#Calculate the probabilities of the test data for each species
y_test = clf.predict_proba(x_test)

#Create CSV
submission = pd.DataFrame(y_test, index=test_ids, columns=encoder.classes_)
submission.to_csv('./submission_svm.csv')
