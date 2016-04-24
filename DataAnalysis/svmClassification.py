import pandas as pd
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt

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
    X = np.array(X)
    y = df[target]
    # Binarize the output
    #y = label_binarize(y, classes=[1,3,4,5])
    #n_classes = y.shape[1]
    #print("n_classes",n_classes)
    print(X.shape)
    print(y.shape)
    # Add noisy features
    random_state = np.random.RandomState(0)
    clf = svm.SVC()
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
    # # train the classifiers
    # Run classifier
    classifier = svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state)
    clf = classifier.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)

    print("y_score",y_score)
    # # Compute Precision-Recall and plot curve
    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # for i in range(n_classes):
    #     precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
    #                                                     y_score[:, i])
    #     average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    #
    # # Compute micro-average ROC curve and ROC area
    # precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    # average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
    #
    # # Plot Precision-Recall curve
    # plt.clf()
    # plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    # plt.legend(loc="lower left")
    # plt.show()
    #
    # # Plot Precision-Recall curve for each class
    # plt.clf()
    # plt.plot(recall["micro"], precision["micro"],
    #      label='micro-average Precision-recall curve (area = {0:0.2f})'
    #            ''.format(average_precision["micro"]))
    # for i in range(n_classes):
    #     plt.plot(recall[i], precision[i],
    #          label='Precision-recall curve of class {0} (area = {1:0.2f})'
    #                ''.format(i, average_precision[i]))
    #
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Extension of Precision-Recall curve to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()
    #



    # # test the classifiers
    # svm_pred = svmClf.predict(X_test)
    # #score
    # sv_R = svmClf.score(X_test, y_test)
    # svm_acc_scores = svm_acc_scores + [sv_R]
    #
    #
    #
    # print "============================================="
    # print " Results of optimization "
    # print "============================================="
    # print "SVM Mean R^2", np.mean(svm_acc_scores)
    # # get support vectors
    # print "get support vectors", svmClf.support_vectors_
    # # get indices of support vectors
    # print "get indices of support vectors", svmClf.support_
    # # get number of support vectors for each class
    # print " get number of support vectors for each class", svmClf.n_support_
