import csv
#library readMe: https://docs.python.org/2/library/csv.html

def read_data(file_path):
	with open('data/student-mat.csv', 'rb') as csvfile:
		datareader = csv.reader(csvfile, delimiter=';', quotechar='|')
		data = []
		for row in datareader:
			data.append(row)
	return data

if __name__ == "__main__":
	data = read_data('data/student-mat.csv')	

