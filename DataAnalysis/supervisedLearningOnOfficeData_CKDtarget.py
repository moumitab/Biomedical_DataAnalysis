#Descision tree with CKD as the target and 40 ffeatures in the office record-set
#with 40 conditions as the features
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


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

    data = pd.read_csv(fileIutput,delimiter=',',na_values = '')

    #The lable visit_type is changed into integers
    df1, newRace = encode_target(data, "Race","Race_categorical")
    df2, newEthnicity = encode_target(df1, "Ethnicity","Ethnicity_categorical")
    #df3, newGender = encode_target(df2, "Gender","Gender_categorical")

    # df3['visit_start_date_time'].apply(lambda x: x.toordinal())
    # print(df3['visit_start_date_time'])

    # Get all the columns from the dataframe.
    columns = list(df2.columns)
    print(columns)
    # Filter the columns to remove ones we don't want.
    columns = [c for c in columns if c not in ["person_id","GFR","HeartRate","RespiratoryRate","BodyTemperature","visit_start_date_time",
                                               "BodyMassIndex","PulseOx","visit_occurrence_id","conditions","ckd","TelephoneConsults","Race","Ethnicity","Gender","RxRefillConsult"]]
    # Store the variable we'll be predicting on.
    target = "ckd"
    df = df2.dropna(subset = [target,'SystolicBp','DiastolicBp','BodyWeight','BodyHeight'])

    X = df[columns]
    y = df[target]
    y = y.reshape(len(y),1)
    #y = pd.reshape(y,1)
    temp_list=[]
    Error=np.zeros((len(columns),1))

    # Generate the training set.  Set random_state to be able to replicate results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

    print(y_train.shape)
    print(X_train.columns)
     ######################################################################
    #DecisionTreeClassifier
    ######################################################################
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    # Make predictions.
    predictions_DT = clf.predict(X_test)
    #predictions_DT = predictions_DT.reshape(len(predictions_DT),1)
    predictions_DT = np.array(predictions_DT)
    print(len(predictions_DT))
    print(predictions_DT)

    ##Selecting the features most important
     #Plot
    print (sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), columns),
             reverse=True))

    importances = clf.feature_importances_

    print("importance",importances)

    indices = np.argsort(importances)[::-1]
    print("indices",indices)
    print("X_train.shape[1]",X_train.shape[1])
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    newCol = 'predictions_DT'
    # Compute the error.
    r = mean_squared_error(predictions_DT, y_test)
    print(abs(r))
    accuracy_of_descisionTree = clf.score(X_test,y_test)
    print(accuracy_of_descisionTree)


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