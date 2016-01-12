import numpy as np
import pandas as pd

import time

rs = 19683
from sklearn import svm, cross_validation, preprocessing, metrics, decomposition
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from kaggletdevries import getDummiesInplace, pdFillNAN, make_dataset, pdStandardScaler

if __name__ == '__main__':
	t0 = time.time()
	print ("Creating dataset...") 
	train, test, labels = make_dataset(useDummies = True, 
	    fillNANStrategy = "mean", useNormalization = True)

	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(train, labels, \
			test_size=0.2, random_state=rs)

	# PCA - this won't help but whatever
	pca = decomposition.PCA(n_components=150)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	# Train the classifier and fit to training data
	# Grid search for RBF
	Cs = np.logspace(0, 4, 5)
	gammas = np.logspace(-4,0,5)
	classifier = GridSearchCV(estimator=svm.SVC(), \
		param_grid=dict(C=Cs,gamma=gammas,kernel=['linear']),
		verbose=3,
		n_jobs=-1,scoring='accuracy' )

	classifier.fit(X_train, y_train)
	print classifier.best_score_
	print classifier.best_estimator_

	t1 = time.time()
	print("training and testing took %f minutes" % ((t1-t0)/60.))
    