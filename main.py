# print(__doc__)

#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

from sklearn import svm, datasets, feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVC, SVR
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE

from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston

import sys
import numpy as np
import csv
# library readme: https://docs.python.org/2/library/csv.html
from sklearn.neural_network import MLPClassifier
# library readme: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

def read_target(file_path):
	with open('data/student-mat.csv', 'rb') as csvfile:
		targetreader = csv.reader(csvfile, delimiter=';')
		target = []
		for row in targetreader:
			target.append(row)
	return target


def prepare_data(entry):
	if entry[0] == 'GP':
		entry[0] = 0
	else:
		entry[0] = 1

	if entry[1] == 'F':
		entry[1] = 0
	else:
		entry[1] = 1

	if entry[3] == 'U':
		entry[3] = 0
	else:
		entry[3] = 1

	if entry[4] == 'LE3':
		entry[4] = 0
	else:
		entry[4] = 1

	if entry[5] == 'T':
		entry[5] = 0
	else:
		entry[5] = 1

	if entry[8] == 'teacher':
		entry[8] = 0
	elif entry[8] == 'health':
		entry[8] = 1
	elif entry[8] == 'services':
		entry[8] = 2
	elif entry[8] == 'at_home':
		entry[8] = 3
	elif entry[8] == 'other':
		entry[8] = 4

	if entry[9] == 'teacher':
		entry[9] = 0
	elif entry[9] == 'health':
		entry[9] = 1
	elif entry[9] == 'services':
		entry[9] = 2
	elif entry[9] == 'at_home':
		entry[9] = 3
	elif entry[9] == 'other':
		entry[9] = 4

	if entry[10] == 'home':
		entry[10] = 0
	elif entry[10] == 'reputation':
		entry[10] = 1
	elif entry[10] == 'course':
		entry[10] = 2
	else:
		entry[10] = 3

	if entry[11] == 'mother':
		entry[11] = 0
	elif entry[11] == 'father':
		entry[11] = 1
	else:
		entry[11] = 2

	if entry[15] == 'yes':
		entry[15] = 1
	else:
		entry[15] = 0

	if entry[16] == 'yes':
		entry[16] = 1
	else:
		entry[16] = 0

	if entry[17] == 'yes':
		entry[17] = 1
	else:
		entry[17] = 0

	if entry[18] == 'yes':
		entry[18] = 1
	else:
		entry[18] = 0

	if entry[19] == 'yes':
		entry[19] = 1
	else:
		entry[19] = 0

	if entry[20] == 'yes':
		entry[20] = 1
	else:
		entry[20] = 0

	if entry[21] == 'yes':
		entry[21] = 1
	else:
		entry[21] = 0

	if entry[22] == 'yes':
		entry[22] = 1
	else:
		entry[22] = 0
	del entry[32]  # to remove target

	return entry


def pipeline_anova_svm(data, target):
	y = target
	X = data

	anova_filter = SelectKBest(f_regression, k=1)
	clf = svm.SVC(kernel='linear')

	anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
	anova_svm.set_params(anova__k=1, svc__C=.1).fit(X, y)

	prediction = anova_svm.predict(X)
	#print(anova_svm.score(X, y))
	#print(anova_svm.named_steps['anova'].score_func(X, y)[1])
	print sorted(zip(map(lambda x: round(x, 4), anova_svm.named_steps['anova'].score_func(X, y)[1]), feature_names), reverse=True)
	#print(anova_svm)
	print(anova_svm.named_steps['anova'].get_support())


def univariate_feature_selection(data, target):
	y = target
	X = data
	n_samples = len(y)
	X = np.reshape(X, (n_samples, -1))
	X = np.hstack((X, 2 * np.random.random((n_samples, 400))))

	transform = feature_selection.SelectPercentile(feature_selection.f_classif)
	clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])

	score_means = list()
	score_stds = list()
	percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

	for percentile in percentiles:
		clf.set_params(anova__percentile=percentile)
    	# Compute cross-validation score using 1 CPU
		this_scores = cross_val_score(clf, X, y, n_jobs=1)
		score_means.append(this_scores.mean())
		score_stds.append(this_scores.std())

	plt.errorbar(percentiles, score_means, np.array(score_stds))
	plt.title('Performance of the SVM-Anova varying the percentile of features selected')
	plt.xlabel('Percentile')
	plt.ylabel('Prediction rate')

	plt.axis('tight')
	plt.show()

def random_forest(data, target, feature_names):
	X = data
	y = target
	rf = RandomForestRegressor()
	rf.fit(X, y)
	print "Features sorted by their score:"
	print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)


def stability_selection(data, target, feature_names):
	X = data
	y = target
	rlasso = RandomizedLasso()
	rlasso.fit(X, y)

	print "Features sorted by their score:"
	print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                 feature_names), reverse=True)



def recursive_feature_elimination(data, target, feature_names):
	X = data
	y = target

	svc = SVC(kernel="linear", C=1)
	rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
	#lr = LinearRegression()
	#rank all features, i.e continue the elimination until the last one
	#rfe = RFE(lr, n_features_to_select=1)
	rfe.fit(X, y)
	print "Features sorted by their rank:"
	print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_names))



if __name__ == "__main__":
	unparsed_data = read_target('target/student-mat.csv')
	feature_names = unparsed_data[0]
	del unparsed_data[0]
	i = 0
	target = []
	data = []
	for Aentry in unparsed_data:
		target.append(Aentry[32])
		data.append(prepare_data(Aentry))
		#print "Entry #",
		#print i
		#for segment in Aentry:
		#	print segment,
		#print ""
		#print "target: ",
		#print target[i]
		i += 1
		for x in range(len(Aentry)):
			Aentry[x] = int(Aentry[x])
	target = np.array(target).astype(np.float)
	#univariate_feature_selection(data, target)
	pipeline_anova_svm(data, target)
	#recursive_feature_elimination(data, target, feature_names)
	stability_selection(data, target, feature_names)
	random_forest(data, target, feature_names)
