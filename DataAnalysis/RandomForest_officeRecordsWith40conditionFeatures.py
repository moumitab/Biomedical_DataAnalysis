#Random forest tree with CKD as the target and 40 conditions as features in the office record-set
#Random forest tree with CKD as the target and next 50 conditions as features in the office record-set
#we repeat this process for 4 times each time with 40-50 different conditions as features
#with 40 conditions as the features
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



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
    fileIutput =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\\updatedData\\OfficeRecords_139TopConditionFeatures_factored_selectedFeatures.csv'

    data = pd.read_csv(fileIutput,delimiter=',')
    data = data.dropna()
    # Get all the columns from the dataframe.
    columns = list(data.columns)
    print(columns)
    # Filter the columns to remove ones we don't want.
    #columns = [c for c in columns if c not in ["person_id","GFR","HeartRate","RespiratoryRate","BodyTemperature","visit_start_date_time",
    #                                          "BodyMassIndex","PulseOx","visit_occurrence_id","conditions","ckd","TelephoneConsults","Race","Ethnicity","Gender","RxRefillConsult"]]

    columns = [c for c in columns if c not in ["ckd"]]

    # Store the variable we'll be predicting on.
    target = "ckd"
    #df = data.dropna(subset = [target,'SystolicBp','DiastolicBp','BodyWeight','BodyHeight'])
    print(data.shape)
    X = data[columns]
    y = data[target]

    #y = y.reshape(len(y),1)
    #y = pd.reshape(y,1)
    temp_list=[]
    Error=np.zeros((len(columns),1))

    # Generate the training set.  Set random_state to be able to replicate results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

    #######################################################################################
    #Train a random forest
    ######################################################################################
    from sklearn.metrics import r2_score
    model = RandomForestClassifier(n_estimators = 10)
    forest = model.fit(X_train, y_train)
    R2Score = r2_score(y_test, forest.predict(X_test))
    print("R^2",R2Score)
    acc = forest.score(X_test,y_test)
    print("accuracy",acc)
    disbursed = forest.predict_proba(X_test)

    print (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), columns),
             reverse=True))
    importances = forest.feature_importances_

    print("importance",importances)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    print("indices",indices)
    print("X_train.shape[1]",X_train.shape[1])
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
