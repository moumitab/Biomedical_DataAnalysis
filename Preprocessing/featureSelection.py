print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import pandas as pd

# Build a classification task using 3 informative features
# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                            n_redundant=2, n_repeated=0, n_classes=8,
#                            n_clusters_per_class=1, random_state=0)

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
X = data[predictor_var]
y= data[outcome_var]
print(X)
print (y)

# Create the RFE object and compute a cross-validated score.
#svc = SVC(kernel="linear")
from sklearn import linear_model
model = linear_model.LogisticRegression(fit_intercept=True, multi_class = "ovr")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
rfecv.fit(X, y)
print('accuracy scoring',rfecv.scoring)
print("Ranking of the features : %d" % rfecv.ranking_)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()