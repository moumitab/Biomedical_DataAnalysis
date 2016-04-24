#This file just maps the SNOMED codes to the description provided in the Data Dictionary
#Input files used are Hospital_Frequency_conditons.csv and SNOMED_CODE.csv
import csv
f = open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Data Dictionary\\SNOMED_CODE.csv')
reader = csv.reader(f)
result = {}
for row  in reader:
    key = row[0]
    if key in result:
        pass
    result[key] = row[1:]


with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Hospital_cond_new.csv','r') as csvinput:
    with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\HospitalConditionsDesc_new_desc.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            if result.get(row[0]):
                description = result.get(row[0])
                writer.writerow(row + description)
            else:
                writer.writerow(row + ['code Not found'])
