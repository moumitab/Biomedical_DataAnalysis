import pandas as pd
from pandas import DataFrame
import datetime
import pandas.io.data
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
#Handling Missing data
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
# Import the linearregression model.
from sklearn.linear_model import LinearRegression
# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error
# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO

from sklearn.ensemble import RandomForestClassifier

def encode_target(df, target_column,columnName):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[columnName] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

if __name__ == '__main__':
    #Input the office records with 30 top conditions as features
    fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\\updatedData\\OfficeRecords_40TopConditionFeatures.csv'
    fileOutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\\updatedData\\DTofOfficeData_withoutMissing.csv'
    DTOutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\DT_tree.dot'
    data = pd.read_csv(fileIutput,delimiter=',',na_values = '')

    #The lable visit_type is changed into integers
    df1, newRace = encode_target(data, "Race","Race_categorical")
    df2, newEthnicity = encode_target(df1, "Ethnicity","Ethnicity_categorical")
    df3, newGender = encode_target(df2, "Gender","Gender_categorical")

    #df3['visit_start_date_time'] = df3[['visit_start_date_time']].astype(datetime)
    #df3['visit_start_date_time'] = df3[['visit_start_date_time']].astype(datetime)

    # Get all the columns from the dataframe.
    columns = list(df3.columns[2:])
    print(columns)
    # Filter the columns to remove ones we don't want.
    columns = [c for c in columns if c not in ["\xef\xbb\xbfperson_id","GFR","HeartRate","RespiratoryRate","BodyTemperature","DiastolicBp",
        "SystolicBp","BodyHeight","BodyWeight","BodyMassIndex","PulseOx","visit_start_date_time","visit_occurrence_id","conditions","ckd","TelephoneConsults","Race","Ethnicity","Gender","RxRefillConsult"]]
    # Store the variable we'll be predicting on.
    target = "ckd"
    df = df3.dropna(subset = [target, 'GFR'])
    print(columns)
    y = np.array(df[target])
    temp_list=[]
    Error=np.zeros((len(columns),1))
    #Plot each of the predictor and the target variable using linear regression
    #This code shows that our predictors are not linearly related to the target variable
    # for i in range(0,len(columns)):
    #     lin_model = LinearRegression(fit_intercept=True)
    #     x=np.array(df[columns[i]])
    #     x = x.reshape(len(x),1)
    #     print("columns[i]",columns[i])
    #     lin_model.fit(x,y)
    #     print("columns[i]",columns[i])
    #     Error[i]=(abs(y-lin_model.predict(x))).mean()
    #     #print("lin_model.predict(x)",lin_model.predict(x))
    #     plt.plot(x,lin_model.predict(x),'r',x,y,'b^')
    #     plt.xlabel('Predictor')
    #     plt.ylabel('Price')
    #     plt.title(columns[i])
    #     plt.show()

    # Generate the training set.  Set random_state to be able to replicate results.
    train = df.sample(frac=0.8, random_state=1)
    # Select anything not in the training set and put it in the testing set.
    test = df.loc[~df.index.isin(train.index)]
    x_train = train[columns]
    print(x_train['Age'])
    x_test = test[columns]
    y_train = train[target]
    y_test = test[target]
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(y_train.shape)
    print(x_train.shape)
     ######################################################################
    #DecisionTreeClassifier
    ######################################################################
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    # Make predictions.
    predictions_DT = clf.predict(df[columns])
    #predictions_DT = predictions_DT.reshape(len(predictions_DT),1)
    predictions_DT = np.array(predictions_DT)
    print(len(predictions_DT))
    print(predictions_DT[0])
    newCol = 'predictions_DT'
    # for i in range(0,(len(predictions_DT)-1)):
    #     df.loc[i,newCol] = predictions_DT[i]
    #df.to_csv(fileOutput)


    from sklearn.externals.six import StringIO
    #from IPython.display import Image
    import StringIO,pydot
    from os import system
    # dotfile = open("C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\dtree2.dot", 'w')
    # out = tree.export_graphviz(clf, out_file = dotfile, feature_names = columns)
    # dotfile.close()
    # #system("dot -Tpng C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\dtree2.dot .dot -o C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\dtree2.dot\\dtree2.png")
    #
    # graph = pydot.graph_from_dot_data((out.getvalue()).write_pdf("officeData_DT.pdf"))

    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    print pydot.__file__
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #Image(graph.create_png())
    graph.write_pdf("officeData_DT.pdf")



    # Compute the error.
    mean_squared_error(predictions_DT, df[target])
    print(abs(mean_squared_error(predictions_DT, df[target])))
    #print(clf.feature_importances_)
    #
    # # Initialize the model class.
    # model = LinearRegression()
    # # Fit the model to the training data.
    # model.fit(train[columns], train[target])
    # # Generate our predictions for the test set.
    # predictions = model.predict(test[columns])
    # # Compute error between our test predictions and the actual values.
    # mean_squared_error(predictions, test[target])
    # print(mean_squared_error(predictions, test[target]))
    #
    # # plt.plot(test[columns],clf.predict(test[columns]),'r',test[columns],test[target])
    # # plt.xlabel('Predictor')
    # # plt.ylabel('target')
    # # plt.title("PLOTS")
    # # plt.show()
    #
    # # Initialize the model with some parameters.
    # model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
    # # Fit the model to the data.
    # model.fit(train[columns], train[target])
    # # Make predictions.
    # predictions = model.predict(test[columns])
    # # Compute the error.
    # mean_squared_error(predictions, test[target])
    # print(abs(mean_squared_error(predictions, test[target])))
    # importances = model.feature_importances_
    # indices = np.argsort(importances)[::-1]
    #     # Print the feature ranking
    # print("Feature ranking:")
    #
    # for f in range(train[columns].shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #     print("columns[indices[f]]",columns[indices[f]])

    #Some code deleted
    # missing_values is the value of your placeholder, strategy is if you'd like mean, median or mode, and axis=0 means it calculates the imputation based on the other feature values for that sample
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #imp  = imp.fit(train)
    #new_train_data=imp.transform(train)
    #print(new_train_data)