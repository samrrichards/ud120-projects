#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# print len(features_train[0])


#########################################################
### your code goes here ###

from sklearn.tree import DecisionTreeClassifier

classify = DecisionTreeClassifier(min_samples_split=40)

t0 = time()
classify.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
classify.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print classify.score(features_test, labels_test)

#########################################################
