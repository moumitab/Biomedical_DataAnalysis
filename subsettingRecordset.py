#In this code we will subset the office and lab records into new record-sets with the
# all the patients present in the hospital records
import pandas as pd
import csv
import numpy as np

#Input the hospital unique patient list
file1 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\uniquePatient_id_hospitalRecords.csv'
file2 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\updatedData\\Office_20160303_sorted_date_2.csv'
file2Out = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\SubDatasets\\Office_20160303_sorted_date_patientHospitalized_subset1.csv'


#Dataframe for each input
uniquePatientIDdata = pd.read_csv(file1)
officeRecordSetData = pd.read_csv(file2)

colnamesOffice = list(officeRecordSetData.columns.values)

#columns = list(data.columns[2:])
print("colnamesOffice",len(colnamesOffice))
#Convert the unique patient list into a dictionary
uniqueID_dict = {}
for i, row in uniquePatientIDdata.iterrows():
    key = row['person_id']
    if key in uniqueID_dict:
        pass
    uniqueID_dict[key] = row['Frequency']
print("result.has_key(3)",uniqueID_dict.has_key(3))

#Subset the office Data set to contain only those patients who have been hospitalized atleast once or more

newOfficeRecordSet = pd.DataFrame(columns=colnamesOffice)
print("officeRecordSetData.columns.values",officeRecordSetData.columns.values)
print("newOfficeRecordSet.colnames",newOfficeRecordSet.columns.values)

j = 0
for i, row in officeRecordSetData.iterrows():
    personID = row['person_id']
    #print("newOfficeRecordSet.loc[[j],'\xef\xbb\xbfperson_id']",newOfficeRecordSet.loc[j,'\xef\xbb\xbfperson_id'])
    if(uniqueID_dict.has_key(personID)):
        newOfficeRecordSet.loc[j,'person_id'] = row['person_id']
        newOfficeRecordSet.loc[j,'visit_occurrence_id'] = row['visit_occurrence_id']
        newOfficeRecordSet.loc[j,'visit_start_date_time'] = row['visit_start_date_time']
        newOfficeRecordSet.loc[j,'Gender'] = row['Gender']
        newOfficeRecordSet.loc[j,'Race'] = row['Race']
        newOfficeRecordSet.loc[j,'Ethnicity'] = row['Ethnicity']
        newOfficeRecordSet.loc[j,'Age'] = row['Age']
        newOfficeRecordSet.loc[j,'conditions'] = row['conditions']
        newOfficeRecordSet.loc[j,'HeartRate'] = row['HeartRate']
        newOfficeRecordSet.loc[j,'RespiratoryRate'] = row['RespiratoryRate']
        newOfficeRecordSet.loc[j,'BodyTemperature'] = row['BodyTemperature']
        newOfficeRecordSet.loc[j,'SystolicBp'] = row['SystolicBp']
        newOfficeRecordSet.loc[j,'DiastolicBp'] = row['DiastolicBp']
        newOfficeRecordSet.loc[j,'PulseOx'] = row['PulseOx']
        newOfficeRecordSet.loc[j,'BodyHeight'] = row['BodyHeight']
        newOfficeRecordSet.loc[j,'BodyWeight'] = row['BodyWeight']
        newOfficeRecordSet.loc[j,'BodyMassIndex'] = row['BodyMassIndex']
        newOfficeRecordSet.loc[j,'GFR'] = row['GFR']
        newOfficeRecordSet.loc[j,'ckd'] = row['ckd']
        newOfficeRecordSet.loc[j,'TelephoneConsults'] = row['TelephoneConsults']
        newOfficeRecordSet.loc[j,'RxRefillConsult'] = row['RxRefillConsult']
        newOfficeRecordSet.loc[j,'ACEInhibitors'] = row['ACEInhibitors']
        newOfficeRecordSet.loc[j,'AldosteroneReceptorAntagonists'] = row['AldosteroneReceptorAntagonists']
        newOfficeRecordSet.loc[j,'AlphaBetaBlockers'] = row['AlphaBetaBlockers']
        newOfficeRecordSet.loc[j,'AnalgesicAntipyreticNonNarcotic'] = row['AnalgesicAntipyreticNonNarcotic']
        newOfficeRecordSet.loc[j,'AngiotensinIIReceptorBlocker'] = row['AngiotensinIIReceptorBlocker']
        newOfficeRecordSet.loc[j,'AntihyperglycemicAlphaGlucosidaseInhibitors'] = row['AntihyperglycemicAlphaGlucosidaseInhibitors']
        newOfficeRecordSet.loc[j,'AntihyperglycemicDipeptidylPeptidase4Inhibitors'] = row['AntihyperglycemicDipeptidylPeptidase4Inhibitors']
        newOfficeRecordSet.loc[j,'AntihyperlipidemicBileAcidSequestrants'] = row['AntihyperlipidemicBileAcidSequestrants']
        newOfficeRecordSet.loc[j,'AntihyperlipidemicFibricAcidDerivatives'] = row['AntihyperlipidemicFibricAcidDerivatives']
        newOfficeRecordSet.loc[j,'AntihyperlipidemicHMGCoAReductaseInhibitors'] = row['AntihyperlipidemicHMGCoAReductaseInhibitors']
        newOfficeRecordSet.loc[j,'AntihyperlipidemicSelectiveCholesterolAbsorptionInhibitor'] = row['AntihyperlipidemicSelectiveCholesterolAbsorptionInhibitor']
        newOfficeRecordSet.loc[j,'AntihyperlipidemicAgentsDietarySource'] = row['AntihyperlipidemicAgentsDietarySource']
        newOfficeRecordSet.loc[j,'BetaBlockersCardiacSelective'] = row['BetaBlockersCardiacSelective']
        newOfficeRecordSet.loc[j,'BetaBlockersNonCardiacSelective'] = row['BetaBlockersNonCardiacSelective']
        newOfficeRecordSet.loc[j,'CalcimimeticParathyroidCalciumReceptorSensitivityEnhancer'] = row['CalcimimeticParathyroidCalciumReceptorSensitivityEnhancer']
        newOfficeRecordSet.loc[j,'CalciumChannelBlockersBenzothiazepinesPhenylakylamines'] = row['CalciumChannelBlockersBenzothiazepinesPhenylakylamines']
        newOfficeRecordSet.loc[j,'CalciumChannelBlockersDihydropyridinesAndOther'] = row['CalciumChannelBlockersDihydropyridinesAndOther']
        newOfficeRecordSet.loc[j,'CentralAlpha2ReceptorAgonists'] = row['CentralAlpha2ReceptorAgonists']
        newOfficeRecordSet.loc[j,'DirectActingVasodilators'] = row['DirectActingVasodilators']
        newOfficeRecordSet.loc[j,'DiureticThiazides'] = row['DiureticThiazides']
        newOfficeRecordSet.loc[j,'InsulinResponseEnhancersNotSpecified'] = row['InsulinResponseEnhancersNotSpecified']
        newOfficeRecordSet.loc[j,'InsulinResponseEnhancersBiguanides'] = row['InsulinResponseEnhancersBiguanides']
        newOfficeRecordSet.loc[j,'InsulinResponseEnhancersThiazolidinediones'] = row['InsulinResponseEnhancersThiazolidinediones']
        newOfficeRecordSet.loc[j,'Insulins'] = row['Insulins']
        newOfficeRecordSet.loc[j,'NSAID'] = row['NSAID']
        newOfficeRecordSet.loc[j,'NSAIDAnalgesicTopical'] = row['NSAIDAnalgesicTopical']
        newOfficeRecordSet.loc[j,'OralAntidiabeticInsulinReleaseStimulantType'] = row['OralAntidiabeticInsulinReleaseStimulantType']
        newOfficeRecordSet.loc[j,'OralAntihyperglycemicMeglitinideAnalogs'] = row['OralAntihyperglycemicMeglitinideAnalogs']
        newOfficeRecordSet.loc[j,'OralAntihyperglycemicSulfonylureaDerivatives'] = row['OralAntihyperglycemicSulfonylureaDerivatives']
        newOfficeRecordSet.loc[j,'PeripheralAlpha1ReceptorBlockers'] = row['PeripheralAlpha1ReceptorBlockers']
        newOfficeRecordSet.loc[j,'ReninInhibitors'] = row['ReninInhibitors']
        newOfficeRecordSet.loc[j,'VitaminsBPreparations'] = row['VitaminsBPreparations']
        newOfficeRecordSet.loc[j,'VitaminsB3'] = row['VitaminsB3']

        j = j +1
        #print("newOfficeRecordSet",newOfficeRecordSet)
        print("j",j)

#print("newOfficeRecordSet",newOfficeRecordSet)
newOfficeRecordSet.to_csv(file2Out)


