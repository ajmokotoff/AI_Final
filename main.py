import csv
#library readMe: https://docs.python.org/2/library/csv.html

with open('data/student-mat.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in datareader:
		print ', '.join(row)
