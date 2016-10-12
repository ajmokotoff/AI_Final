import csv

import numpy
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

# library readMe: https://docs.python.org/2/library/csv.html

def read_data(file_path):
    with open(file_path, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=';')
        data = []
        for row in datareader:
            data.append(row)
    return data

def all_strings_to_num(entry,delCat):
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

    del entry[-3:]
    if delCat == 'general':
        del entry[:3] #delete school, sex, age, and address
    elif delCat == 'family':
        del entry[23]  # delete famrel
        del entry[12]  # delete guardian
        del entry[4:9] #delete famsize, Pstatus, Medu, Fedu, Mjob, and Fjob
    elif delCat == 'school':
        del entry[29]  # delete absences
        del entry[13:20] #delete studytime, failures, schoolsup, famsup, paid, activities, nursery, and higher
        del entry[10] #delete reason
    elif delCat == 'personal':
        del entry[24:28] #delete freetime, goout, Dalc, Walc, and health
        del entry[21:22] #delete internet, and romantic

if __name__ == "__main__":
    if len(sys.argv) > 1:
        delCat = str(sys.argv[1])
    else:
        delCat = ''

    data = read_data('data/student-mat.csv')
    del data[0]
    i = 0
    targets = []
    for Aentry in data:
        targets.append(Aentry[32])
        all_strings_to_num(Aentry,delCat)
        print "Entry #",
        print i
        for segment in Aentry:
            print segment,
        print ""
        print "Target: ",
        print targets[i]
        i += 1
        for x in range(len(Aentry)):
            Aentry[x] = int(Aentry[x])

    # Train algorithm on first 300 data entries
    # rf = RandomForestRegressor()
    # rf.fit(data[:300], targets[:300])

    # Change the value of instanceNumber to change which data to use for test
    # Keep value above 300 so as to avoid testing on training data
    instanceNumber = 301
    # print "Instance Prediction: ", rf.predict(data[instanceNumber])
    # print "Actual Score: ", targets[instanceNumber]

    # Train Boosted Stump Algorithm
    # maxScore = 0
    # maxEst = 0
    # for i in range(1,100):
    clf = AdaBoostClassifier(n_estimators=24,algorithm="SAMME")
    # scores = cross_val_score(clf, data[:300], targets[:300])
    clf.fit(data[:300],targets[:300])
    # print "Boosted Stump Prediction: ", scores.mean()
    predicted = []
    predicted = clf.predict(data[instanceNumber:])
    predList = predicted.tolist()
    targList = targets[instanceNumber:]
    error = []
    for i in range(0,len(data)-instanceNumber):
        error.append(int(predList[i]) - int(targList[i]))
    print "Boosted Stump Prediction: " + str(predicted)
    print "Actual Score: " + str(targets[instanceNumber:])
    print "Error: " + str(error)
    print
    score = clf.score(data[instanceNumber:],targets[instanceNumber:])
    # if score > maxScore:
    #     maxScore = score
    #     maxEst = i
    print "Score: " + str(score)
    # print "maxScore =  " + str(maxScore)
    # print "max estimators = " + str(maxEst)
