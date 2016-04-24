#Here what we do is just replace the empty cells with 0 in any .csv file and write the results in a new file

import pandas as pd
import numpy as np
conditionVec =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\Condtions_Vectors_Hoss_Input.csv'
dataFrame = pd.read_csv(conditionVec)
# arr = np.array(data)
# arr[np.where(arr==1)] = '2'
# arr[np.where(arr==0)] = '1'
# dataFrame = pd.DataFrame(arr)
dataFrame.replace(to_replace = 1 ,value  = 2,inplace = True)
dataFrame.replace(to_replace = 0 ,value  = 1,inplace = True)
conditionVec =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\Condtions_Vectors_Hoss_Input_2.csv'
dataFrame.to_csv(conditionVec)
