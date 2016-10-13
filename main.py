# print(__doc__)

#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC, SVR
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import (RandomizedLasso, LinearRegression, Ridge, Lasso)

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


def gauss_naive_bayes(option, opt, value, parser):
	 # Train Gaussian Naive Bayes
	predictFrom = 301
	gnb = GaussianNB()
	gnb.fit(X[:300], y[:300])
	predicted = gnb.predict(X[predictFrom:])
	predList = predicted.tolist()
	targList = y[predictFrom:]
	error = []
	score = gnb.score(X[predictFrom:],y[predictFrom:])

	for i in range(0,len(X)-predictFrom):
		error.append(int(predList[i]) - int(targList[i]))
	print "\nGaussian Naive Bayes Prediction: " + str(predicted)
	print "Actual Score: " + str(y[predictFrom:])
	print "Error: " + str(error)
	print "Score: " + str(score)
	print


def multi_naive_bayes(option, opt, value, parser):
	# Train Multinomial Naive Bayes
	predictFrom = 301
	mnb = MultinomialNB()
	mnb.fit(X[:300], y[:300])
	predicted = mnb.predict(X[predictFrom:])
	predList = predicted.tolist()
	targList = targets[predictFrom:]
	error = []
	score = mnb.score(X[predictFrom:],y[predictFrom:])

	for i in range(0,len(X)-predictFrom):
		error.append(int(predList[i]) - int(targList[i]))
	print "Multi Naive Bayes Prediction: " + str(predicted)
	print "Actual Score: " + str(y[predictFrom:])
	print "Error: " + str(error)
	print "Score: " + str(score)
	print


def bern_naive_bayes(option, opt, value, parser):
	 # Train Bernoulli Naive Bayes
	predictFrom = 301
	bnb = BernoulliNB()
	bnb.fit(X[:300], y[:300])
	predicted = bnb.predict(data[predictFrom:])
	predList = predicted.tolist()
	targList = y[predictFrom:]
	error = []
	score = bnb.score(X[predictFrom:],y[predictFrom:])

	for i in range(0,len(X)-predictFrom):
		error.append(int(predList[i]) - int(targList[i]))
	print "\Bern Naive Bayes Prediction: " + str(predicted)
	print "Actual Score: " + str(y[predictFrom:])
	print "Error: " + str(error)
	print "Score: " + str(score)
	print


def pipeline_anova_svm(option, opt, value, parser):
	anova_filter = SelectKBest(f_regression, k=1)
	clf = svm.SVC(kernel='linear')

	anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
	anova_svm.set_params(anova__k=1, svc__C=.1).fit(X, y)

	prediction = anova_svm.predict(X)
	#print(anova_svm.score(X, y))
	#print(anova_svm.named_steps['anova'].score_func(X, y)[1])
	print "\nPipeline Anova SVM: Features sorted by rank:"
	print sorted(zip(map(lambda x: round(x, 4), anova_svm.named_steps['anova'].score_func(X, y)[1]), feature_names), reverse=True)
	#print(anova_svm)
	print(anova_svm.named_steps['anova'].get_support())
	print


def univariate_feature_selection(option, opt, value, parser):
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

def random_forest(option, opt, value, parser):
	rf = RandomForestRegressor()
	rf.fit(X, y)
	print "\nRandom Forest: Features sorted by rank:"
	print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)
	print


def stability_selection(option, opt, value, parser):
	rlasso = RandomizedLasso()
	rlasso.fit(X, y)

	print "\nStability Selection: Features sorted by rank:"
	print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
				 feature_names), reverse=True)
	print



def recursive_feature_elimination(option, opt, value, parser):
	svc = SVC(kernel="linear", C=1)
	rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
	#lr = LinearRegression()
	#rank all features, i.e continue the elimination until the last one
	#rfe = RFE(lr, n_features_to_select=1)
	rfe.fit(X, y)
	print "\nRecurisve Feature Elimination: Features sorted by rank:"
	print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_names))
	print


def ridge_regression(option, opt, value, parser):
	ridge = Ridge(alpha=7)
	ridge.fit(X, y)
	print "\nRidge Regression: Features sorted by rank:"
	print sorted(zip(map(lambda x: round(x, 4), ridge.coef_), feature_names), reverse=True)
	print


def linear_regression(option, opt, value, parser):
	lr = LinearRegression(normalize=True)
	lr.fit(X, y)
	print "\nLinear Regression: features sorted by rank:"
	print sorted(zip(map(lambda x: round(x, 4), lr.coef_), feature_names), reverse=True)
	print


def boosted_stump(option, opt, value, parser):
	clf = AdaBoostClassifier(n_estimators=24,algorithm="SAMME")
	clf.fit(X[:300],y[:300])
	predicted = []
	predicted = clf.predict(X[301:])
	predList = predicted.tolist()
	targList = y[301:]
	error = []
	for i in range(0,len(X)-301):
		error.append(int(predList[i]) - int(targList[i]))
	score = clf.score(X[301:],y[301:])

	print "\nBoosted Stump Prediction: " + str(predicted)
	print "Actual Score: " + str(y[301:])
	print "Error: " + str(error)
	print "Score: " + str(score)
	print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True)
	print





def commandline_menu():
	parser = OptionParser()
	parser.add_option("-p", "--bsp", dest="bsp", action="callback", callback=boosted_stump, help="Boosted Stump")
	parser.add_option("-b", "--gnb", dest="gnb", action="callback", callback=gauss_naive_bayes, help="Gauss Naive Bayes")
	parser.add_option("-m", "--mnb", dest="mnb", action="callback", callback=multi_naive_bayes, help="Multi Naive Bayes")
	parser.add_option("-n", "--bnb", dest="bnb", action="callback", callback=bern_naive_bayes, help="Bern Naive Bayes")
	parser.add_option("-u", "--ufs", dest="ufs", action="callback", callback=univariate_feature_selection, help="Univariate Feature Selection")
	parser.add_option("-s", "--svm", dest="svm", action="callback", callback=pipeline_anova_svm, help="Pipeline Anova SVM")
	parser.add_option("-r", "--rfe", dest="rfe", action="callback", callback=recursive_feature_elimination, help="Recursive Feature Elimination")
	parser.add_option("-e", "--ss", dest="ss", action="callback", callback=stability_selection, help="Stability Selection")
	parser.add_option("-f", "--rf", dest="rf", action="callback", callback=random_forest, help="Random Forest")
	parser.add_option("-g", "--rr", dest="rr", action="callback", callback=ridge_regression, help="Ridge Regression")
	parser.add_option("-l", "--lr", dest="lr", action="callback", callback=linear_regression, help="Linear Regression")
	(options, args) = parser.parse_args()

	if len(args) < 0:
		parser.error("wrong number of arguments, -h for help")


if __name__ == "__main__":
	unparsed_data = read_target('target/student-mat.csv')
	global feature_names
	feature_names = unparsed_data[0]
	del unparsed_data[0]
	i = 0
	target = []
	data = []
	for Aentry in unparsed_data:
		target.append(Aentry[32])
		data.append(prepare_data(Aentry))
		i += 1
		for x in range(len(Aentry)):
			Aentry[x] = int(Aentry[x])
	target = np.array(target).astype(np.float)
	global X
	X = data
	global y
	y = target

	commandline_menu()
