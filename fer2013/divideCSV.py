import csv
import os

fer2013Path = './fer2013.csv'
fer2013TrainPath = './fer2013_train.csv'
fer2013ValPath = './fer2013_val.csv'
fer2013TestPath = './fer2013_test.csv'

with open(fer2013Path) as f:
    csvReader = csv.reader(f)
    header = next(csvReader)
    rows = [row for row in csvReader]

    train = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(fer2013TrainPath, 'w+'),
               lineterminator='\n').writerows([header[:-1]] + train)
    print(len(train))

    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    csv.writer(open(fer2013ValPath, 'w+'),
               lineterminator='\n').writerows([header[:-1]] + val)
    print(len(val))

    test = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(fer2013TestPath, 'w+'),
               lineterminator='\n').writerows([header[:-1]] + test)
    print(len(test))
