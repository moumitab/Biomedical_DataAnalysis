from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)
  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])

    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]

    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)

    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])
  return model


def getData():
    #import data here
    #return the x and y variables
    #fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\updatedData\\trainTestValidate_CKD\\officeRecords_CKD.csv'
    #fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\updatedData\\trainTestValidate_CKD\\officeRecords_CKD.csv'
    fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures_factored.csv'
    df = pd.read_csv(fileIutput,delimiter=',',na_values = '')
    columns = list(df.columns)
    columns = [c for c in columns if c not in ["person_id","visit_occurrence_id","visit_end_date_time","visit_start_date_time","conditions"]]
    data = df[columns]
    data = data.dropna()
    print data.shape
    outcome_var = 'visit_type'
    predictor_var = [c for c in columns if c not in ["visit_type"]]
    #df1 = df.dropna(subset = ['Gender','Race','Ethnicity','Age','SystolicBp','DiastolicBp','BodyHeight','BodyWeight',outcome_var])
    print predictor_var,outcome_var
    return data,predictor_var,outcome_var

def feature_ranking(model,predictor_var):
    #After fitting the model we want to rank the features and their importance for prediction
    print (sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), predictor_var),
             reverse=True))
    #featimp = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), predictor_var),
    #         reverse=True)
    #Create a series with feature importances:
    #featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
    #print featimp

def performance_measurement(y_true,y_predict):
    metric1 = metrics.precision_recall_curve
    metric2 = metrics.auc
    metric3 = metrics.classification_report
    metric4 = metrics.confusion_matrix
    #precision, recall, thresholds = metric1(y_true, y_predict)
    #metric4(y_true, y_predict)
    print(metric4(y_true, y_predict))
    target_names = ['0', '1', '2', '3', '4']
    print(metric3(y_true, y_predict, target_names=target_names))




if __name__ == '__main__':
    data,predictor_var,outcome_var = getData()
    # Generate the training set.  Set random_state to be able to replicate results.
    #X_train, X_test, y_train, y_test = train_test_split(data[predictor_var], data[outcome_var], test_size=0.25,random_state=42)
    train, test = train_test_split(data, test_size=0.25,random_state=42)
    print(train.shape)
    print(test.shape)
    print(train[predictor_var].shape)
    print(train[outcome_var].shape)
    #Decsision tree classification
    #model = DecisionTreeClassifier()
    #Naive Bayes Classifier
    #Naive bayes
    from sklearn.naive_bayes import GaussianNB
    #model = GaussianNB()
    #Logistic Regression
    import matplotlib.pyplot as plt
    #from sklearn import linear_model
    #model = linear_model.LogisticRegression(fit_intercept=True, multi_class = "ovr")
    #Support Vector Machine
    #from sklearn.svm import SVC
    #model = SVC(kernel ='rbf',decision_function_shape = 'ovr',random_state=0)
    #Random Forest Classifier
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    model = RandomForestClassifier(n_estimators=10)
    #estimators = [('reduce_dim', PCA()), ('RandomForestClassifier', RandomForestClassifier())]
    #model = Pipeline(estimators)
    #Lasso and ridge regression
    # prepare a range of alpha values to test
    from sklearn.linear_model import Ridge
    from sklearn.grid_search import GridSearchCV
    #alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    # create and fit a ridge regression model, testing each alpha
    # model = Ridge()
    # grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    # grid.fit(data[predictor_var],data[outcome_var])
    # print(grid)
    # # summarize the results of the grid search
    # print(grid.best_score_)
    # print(grid.best_estimator_.alpha)
    # X = data[predictor_var]
    # alphas = np.array([1])
    # grid_final = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    # grid_final.fit(data[predictor_var],data[outcome_var])
    # grid_final.predict(data[predictor_var])
    # print(grid_final.score())
    # print(grid_final.best_score_)
    # print(grid_final.grid_scores_)

    model = classification_model(model, train,predictor_var,outcome_var)
    feature_ranking(model,predictor_var)
    y_predict = model.predict(test[predictor_var])
    y_true = test[outcome_var]
    print("accuracy from absolutely unseen data",metrics.accuracy_score(y_true,y_predict))
    performance_measurement(y_true,y_predict)
