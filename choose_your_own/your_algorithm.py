#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################

# I built an AdaBoost classifier using SVC estimators.
# I tested it with some other algorithms, but this gave me the best results.

# The accuracy is around 92.8%... not quite the 93.6% Sebastian and Katie report,
# but still pretty close!

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

classify = AdaBoostClassifier(base_estimator=SVC(C=100000), n_estimators=100, algorithm='SAMME')

print "Starting to train algorithm!"

t0 = time()
classify.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
classify.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print classify.score(features_test, labels_test)

# try:
#     prettyPicture(classify, features_test, labels_test)
# except NameError:
#     pass
