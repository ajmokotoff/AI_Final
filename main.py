import csv
# library readme: https://docs.python.org/2/library/csv.html
from sklearn.neural_network import MLPClassifier
# library readme: http://scikit-learn.org/stable/modules/neural_networks_supervised.html 

def read_data(file_path):
	with open('data/student-mat.csv', 'rb') as csvfile:
		datareader = csv.reader(csvfile, delimiter=';', quotechar='|')
		data = []
		for row in datareader:
			data.append(row)
	return data


def multi_layer_perceptron(data):
	X = [[0., 0.], [1., 1.]]
	y = [0, 1]
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	
	print(clf.fit(X, y))
	print(clf.predict([[2., 2.], [-1., -2.]]))


if __name__ == "__main__":
	data = read_data('data/student-mat.csv')
	multi_layer_perceptron(data)	

