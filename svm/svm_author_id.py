#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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

#shortening training sets to speed up the training process

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

from sklearn.svm import SVC

# c_params = [10.0, 100.0, 1000.0, 10000.0]
#
# for param in c_params:
#     classify = SVC(C=param, kernel="rbf")
#
#     print "C param is: " + str(param)
#     t0 = time()
#     print 'starting to train SVC'
#     classify.fit(features_train, labels_train)
#     print "training time:", round(time()-t0, 3), "s"
#
#     t1 = time()
#     classify.predict(features_test)
#     print "prediction time:", round(time()-t1, 3), "s"
#
#     print classify.score(features_test, labels_test)

classify = SVC(C=10000.0, kernel="rbf")

t0 = time()
print 'starting to train SVC'
classify.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = classify.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

# print "prediction for element 10: " + str(pred[10])
# print "prediction for element 26: " + str(pred[26])
# print "prediction for element 50: " + str(pred[50])

print classify.score(features_test, labels_test)

def equals_one(num):
    return num == 1

print 'Chris emails: ' + str(len(filter(equals_one, pred)))


#########################################################
