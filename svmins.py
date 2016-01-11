import numpy as np
import pandas as pd

import time

rs = 19683
from sklearn import svm, cross_validation, preprocessing, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from kaggletdevries import getDummiesInplace, pdFillNAN, make_dataset, pdStandardScaler

if __name__ == '__main__':
	t0 = time.time()
	print ("Creating dataset...") 
    train, test, labels = make_dataset(useDummies = True, 
        fillNANStrategy = "mean", useNormalization = True)

	# Train the classifier and fit to training data
	# Grid search for RBF
	Cs = np.logspace(0, 4, 5)
	gammas = np.logspace(-4,0,5)
	classifier = GridSearchCV(estimator=svm.SVC(), \
		param_grid=dict(C=Cs,gamma=gammas,kernel=['rbf']),
		verbose=32,
		n_jobs=-1,scoring='accuracy' )

	classifier.fit(X_train, y_train)
	print classifier.best_score_
	print classifier.best_estimator_

	t1 = time.time()
	print("training and testing took %f minutes" % ((t1-t0)/60.))
    