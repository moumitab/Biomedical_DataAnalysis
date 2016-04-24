import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter
from datetime import datetime

dataHospital = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\smaller_task\\Hospital_date_dict_subset_135.csv'
dataOffice = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\smaller_task\\Office_date_dict_subset_135.csv'
dataLab = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\smaller_task\\Lab_date_dict_subset_135.csv'

dfHospital = pd.read_csv(dataHospital)
dfOffice = pd.read_csv(dataOffice)
dfLab = pd.read_csv(dataLab)

dict_hospital = defaultdict(list)
dict_office = defaultdict(list)
dict_lab = defaultdict(list)

#Using a dictionary which contains a list of dates for each key
for i, row in dfHospital.iterrows():
    key = row['person_id']
    date_hospital = row['visit_start_date_time']
    if key in dict_hospital:
        dict_hospital[key].append(date_hospital)
    else:
        dict_hospital[key].append(date_hospital)

#Using a dictionary which contains a list of dates for each key
for i, row in dfOffice.iterrows():
    key = row['person_id']
    date_office = row['visit_start_date_time']
    if key in dict_office:
        dict_office[key].append(date_office)
    else:
        dict_office[key].append(date_office)

#Using a dictionary which contains a list of dates for each key
for i, row in dfLab.iterrows():
    key = row['person_id']
    date_lab = row['lab_report_date']
    if key in dict_lab:
        dict_lab[key].append(date_lab)
    else:
        dict_lab[key].append(date_lab)

print(len(dict_hospital))
print(len(dict_office))
print(len(dict_lab))
#Now create a dictionary with key as the pid and value as a list of arrays
#Value should have the date and corresponding visit type as H,O,L
offHospitalLabList = list()

aggregatedList = defaultdict(list)
for eachPatient in dict_hospital:
    listHospital = dict_hospital[eachPatient]
    listOffice = dict_office[eachPatient]
    listLab = dict_lab[eachPatient]
    #listHospitalEach = listHospital.split(',')
    listH = list()
    listHosp = list()
    for eachItem in listHospital:
        #eachItem = datetime.strptime(eachItem,'%Y%m%d')
        listH.append(eachItem)
        listH.append("H")
        listHosp.append(listH)
        listH = list()

    listO = list()
    listOff = list()
    for eachItem in listOffice:
        listO.append(eachItem)
        listO.append("O")
        listOff.append(listO)
        listO = list()


    listL = list()
    listLabtest = list()
    for eachItem in listLab:
        listL.append(eachItem)
        listL.append("L")
        listLabtest.append(listL)
        listL = list()

    #merge sort of office and hospital
    offHospitalList = listHosp + listOff + listLabtest
    key_function = lambda x: datetime.strptime(itemgetter(0)(x), '%m/%d/%Y')
    offHospitalLabList_temp = sorted(offHospitalList, key=key_function, reverse=True)
    offHospitalLabList.append(eachPatient)
    offHospitalLabList.append(offHospitalLabList_temp)
    offHospitalList = list()
    offHospitalLabList_temp = list()

print(offHospitalLabList)
w1 = open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\smaller_task\\TimeLineAnalysisSubset1.csv','w')
print("offHospitalLabList[0]",offHospitalLabList[0])
print("offHospitalLabList[1]",offHospitalLabList[1])
print("offHospitalLabList[2]",offHospitalLabList[2])
print("offHospitalLabList[3]",offHospitalLabList[3])
print("offHospitalLabList[4]",offHospitalLabList[4])
print("offHospitalLabList[5]",offHospitalLabList[5])
print("len(offHospitalLabList)",len(offHospitalLabList))
i = 0
for i in range(len(offHospitalLabList)):
    if((i%2) == 0 ):
        w1.write(str(offHospitalLabList[i]))
        w1.write(',')
    else:
        listDates = offHospitalLabList[i]
        #listDateEach = listDates.lstrip('[').rstrip(']')
        #listDateEach = listDates.split(',')
        j = 0
        for j in range(len(listDates)):
            eachElem = listDates[j]
            for item in eachElem:
                w1.write(str(item))
                w1.write(',')
            w1.write('\n')
            j = j + 1
    print("i",i)
