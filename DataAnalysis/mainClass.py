import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier


if __name__ == '__main__':
    #Input the office records with 30 top conditions as features
    fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\OfficeRecords_30TopConditionFeatures.csv'
    data = pd.read_csv(fileIutput,delimiter=',')

    #converting the data types to the desired one
    data['visit_start_date_time'] = pd.to_datetime(data['visit_start_date_time'])
    data['Gender'] = data['Gender'].astype('category')
    data['Race'] = data['Race'].astype('category')
    data['Ethnicity'] = data['Ethnicity'].astype('category')
    data['RxRefillConsult'] = pd.to_datetime(data['RxRefillConsult'])

    # Get all the columns from the dataframe
    columns = list(data.columns[2:])
    columns = [c for c in columns if c not in ["\xef\xbb\xbfperson_id","HeartRate","RespiratoryRate","BodyTemperature","DiastolicBp",
        "SystolicBp","BodyHeight","BodyWeight","BodyMassIndex","PulseOx","Race","RxRefillConsult","Gender","Ethnicity","GFR","visit_start_date_time","visit_occurrence_id","conditions","ckd","TelephoneConsults"]]
    # Store the variable we'll be predicting on.
    target = "ckd"
    df = data.dropna(subset = [target, 'GFR'])
    n = len(df)
    X = df[columns]
    X = np.matrix(X)
    y = df[target]
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    #Dummy classifier - how the data does by chance
    dc = DummyClassifier(strategy='most_frequent',random_state=0)
    #Gaussian Naive Bayes
    gnb = GaussianNB()
    #SVM
    clf = svm.SVC()
    #10 fold cross-validation
    kf = cross_validation.StratifiedKFold(y, 10)
    gnb_acc_scores = list()
    dc_acc_scores = list()
    svm_acc_scores = list()
    # loop through the folds
    for train, test in kf:

        # extract the train and test sets
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        # train the classifiers
        dc = dc.fit(X_train, y_train)
        gnb = gnb.fit(X_train, y_train)
        svmClf = clf.fit(X_train, y_train)

        # test the classifiers
        dc_pred = dc.predict(X_test)
        gnb_pred = gnb.predict(X_test)
        svm_pred = svmClf.predict(X_test)


        #score
        dc_R = dc.score(X_test, y_test)
        dc_acc_scores = dc_acc_scores + [dc_R]
        gnb_R = gnb.score(X_test, y_test)
        gnb_acc_scores = gnb_acc_scores + [gnb_R]
        sv_R = svmClf.score(X_test, y_test)
        svm_acc_scores = svm_acc_scores + [sv_R]

        #dc_accuracy = mt.accuracy_score(y_test, dc_pred)
        #gnb_accuracy = mt.accuracy_score(y_test, gnb_pred)
        #gnb_acc_scores = gnb_acc_scores + [gnb_accuracy]
        #dc_acc_scores = dc_acc_scores + [dc_accuracy]


    print "============================================="
    print " Results of optimization "
    print "============================================="
    print "Dummy Mean R^2: ", np.mean(dc_acc_scores)
    print "Naive Bayes Mean R^2: ", np.mean(gnb_acc_scores)
    print "SVM Mean R^2", np.mean(svm_acc_scores)

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
    # clf = svm.SVC()
    # clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    # print(clf.score(X_test, y_test))

       # N = 10
    # K = 10 # K-fold CV
    # scores = np.zeros((N,K))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
    # kf = KFold(n, n_folds=K)
    #
    # for i in range(N):
    #     clf = svm.SVC(kernel='rbf')
    #     for train, test in kf:
    #         for j, (train, test) in enumerate(kf):
    #             X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    #             clf.fit(X_train,y_train)
    #             scores[i,j] = clf.score(X_test, y_test)
    # # Compute average CV score for each parameter
    # scores_avg = scores.mean(axis=1)
