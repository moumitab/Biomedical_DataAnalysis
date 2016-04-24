from __future__ import print_function
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict

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

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def selectKImportance(model, X, k=5):
     return X[:,model.feature_importances_.argsort()[::-1][:k]]

if __name__ == '__main__':
    fileInput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\preprocessed\\Hospital_ConditonFeatures_2.csv'
    data = pd.read_csv(fileInput,index_col = 'visit_occurrence_id', parse_dates = True)
    df = pd.DataFrame(data)

    #The lable visit_type is changed into integers
    df2, targets = encode_target(df, "visit_type","Target")
    df3, newRace = encode_target(df2, "Race","Race_categorical")
    df4, newEthnicity = encode_target(df3, "Ethnicity","Ethnicity_categorical")

    #Reordering the colums to have the class lable (visit_type)as the left most column

    sequence = ['visit_start_date_time','visit_end_date_time','LengthOfStayHours','Gender','Race_categorical',
                'Ethnicity_categorical','Age','235595009','44054006','59621000','55822004','194774006','42343007',
                '49436004','40930008','Target','visit_type','Race','Ethnicity','conditions','visit_occurrence_id','person_id']
    df4 = df4.reindex(columns=sequence)

    #Split the data set to train and validate
    df4['is_train'] = np.random.uniform(0, 1, len(df4)) <= .75
    df4, test = df4[df4['is_train']==True], df4[df4['is_train']==False]
    #Create the feature set
    features_train = list(df4.columns[2:15])
    features_test = list(test.columns[2:15])
    y_train = df4["Target"]
    X_train = df4[features_train]
    y_test = test["Target"]
    X_test = test[features_test]

    # Create the random forest object which will include all the parameters
    # for the fit
    model = RandomForestClassifier(n_estimators = 10)

    # Fit the training data to the Survived labels and create the decision trees
    # Train the model using the training sets and check score
    forest = model.fit(X_train, y_train)
    #newX = selectKImportance(forest,X_train,2)
    #print("newX.shape",newX.shape)
    #print("X_train.shape",X_train.shape)
    acc = r2_score(y_test, forest.predict(X_test))
    print("accuracy",acc)
   #Predict Output
    #predicted= forest.predict(X_test)

    disbursed = forest.predict_proba(X_test)
    #fpr, tpr, _ = roc_curve(y_test, disbursed[:,1])
    #roc_auc = auc(fpr, tpr)
    names = ['LengthOfStayHours','Gender','Race_categorical',
             'Ethnicity_categorical','Age','235595009','44054006','59621000','55822004','194774006','42343007',
             '49436004','40930008']
    print (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), features_train),
             reverse=True))

    #Plot
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

    #Plot the feature importances of the forest
    print("features_train",features_train)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    listFeatures = []
    for eachi in indices:
        listFeatures.append(features_train[eachi])
    print(listFeatures)
    plt.xticks(range(X_train.shape[1]), listFeatures )
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    #dt = DecisionTreeClassifier(min_samples_split=2000, random_state=99)
    #dt.fit(X, y)
    #print(dt.feature_importances_)
    #visualize_tree(dt, features)



