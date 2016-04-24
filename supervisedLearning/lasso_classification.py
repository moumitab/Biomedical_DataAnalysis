from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures_factored.csv'
df = pd.read_csv(fileIutput,delimiter=',',na_values = '')
columns = list(df.columns)
columns = [c for c in columns if c not in ["person_id","visit_occurrence_id","visit_end_date_time","visit_start_date_time","conditions"]]
data = df[columns]
data = data.dropna()
outcome_var = 'visit_type'
predictor_var = [c for c in columns if c not in ["visit_type"]]
X = data[predictor_var]
X =  np.array(X)
y = data[outcome_var]
y = np.array(y)
n = len(y)
print n
p = 101
K = 10  # K-fold CV
y = y.reshape(n)

alphas = np.exp(np.linspace(np.log(0.01),np.log(1),100))  # Using log-scale
N = len(alphas) # Number of lasso parameters

scores = np.zeros(N)
alpha = np.zeros(N)
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = LassoCV(n_alphas = 100, cv = K)
    clf = clf.fit(X_train,y_train)
    scores[i] = clf.score(X_test,y_test)
    alpha[i] = clf.alpha_

scores = np.asarray(scores)
max_score_index = np.argmax(scores)
best_alpha = alpha[max_score_index]

print(best_alpha)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = Lasso(alpha=best_alpha)
#clf = LassoCV(n_alphas = 100, cv = K, precompute='auto', n_jobs=2, normalize='True')
clf = clf.fit(X_train,y_train)
scores = clf.score(X_test,y_test)
print(predictor_var[0])
print("clf.coef_",clf.coef_)
i = 0
ind = list()
for i in range(0,len(clf.coef_)):
    if(clf.coef_[i] > 0):
        ind.append(i)

for j in ind:
    print(predictor_var[j])


scores_avg = scores.mean()
scores_std = scores.std()
plt.plot(alphas, scores_avg,'-b')
plt.fill_between(alphas, scores_avg-scores_std, scores_avg+scores_std,facecolor='r',alpha=0.5)
plt.legend([r'Average $R^2$', r'One sd interval'], loc = 'lower left')
plt.plot(alphas, np.ones((len(alphas),1))*scores_avg.max(),'--k', linewidth=1.2)
plt.xlabel(r'$\alpha$', fontsize=18)
plt.ylabel(r'$R^2$', fontsize = 18)


plt.plot(alphas, scores_avg)
plt.xlabel(r'$\alpha$', fontsize=18)
plt.ylabel(r'$R^2$', fontsize = 18)
plt.show()





