from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\updatedData\\trainTestValidate_CKD\\OfficeRecords_139TopConditionFeatures_factored_all_CKD3,4,5.csv'
df = pd.read_csv(fileIutput,delimiter=',',na_values = '')
columns = list(df.columns)
columns = [c for c in columns if c not in ["person_id","visit_occurrence_id","visit_start_date_time","conditions","HeartRate","RespiratoryRate","BodyTemperature",
                                               "PulseOx","BodyMassIndex","GFR","TelephoneConsults","RxRefillConsult"]]
data = df[columns]
data = data.dropna()
print data.shape
outcome_var = 'ckd'
predictor_var = [c for c in columns if c not in ["ckd"]]
X, y = data[predictor_var], data[outcome_var]

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))