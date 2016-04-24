#creating frequency table of conditions in Office data

import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


f = open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\OfficeDataConditions.csv','r')
w1 = open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Office_Frequency_conditons.csv','w')
data = f.readlines()
dict_x = dict()
for dataFirstLine in data:
    dataFirstLine1 = dataFirstLine.strip("'").strip('"').strip('\n').strip('"')
    dataFirstLineEach = dataFirstLine1.split(',')
    for eachItem in dataFirstLineEach:
        if eachItem not in dict_x:
            dict_x[eachItem] =1
        else:
            dict_x[eachItem] = dict_x[eachItem] + 1

for each in dict_x.items():
    w1.write(str(each))
    w1.write('\n')




