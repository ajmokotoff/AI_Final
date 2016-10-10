import csv
#library readMe: https://docs.python.org/2/library/csv.html

with open('data/student-mat.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spamreader:
		print ', '.join(row)
