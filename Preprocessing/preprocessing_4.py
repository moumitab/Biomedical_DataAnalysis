import pandas

onlyConditions = pandas.read_csv("C://ProgramData//MySQL//MySQL Server 5.7//Uploads//conditions_ONLY1.csv",keep_default_na=False, na_values=[""])
firstTwoColums = pandas.read_csv("C://ProgramData//MySQL//MySQL Server 5.7//Uploads//conditions_data2.csv",keep_default_na=False, na_values=[""])

# place the DataFrames side by side
horizontalStack = pandas.concat([firstTwoColums,onlyConditions], axis=1)
# Write DataFrame to CSV
horizontalStack.to_csv('C://ProgramData//MySQL//MySQL Server 5.7//Uploads//outCnditions.csv')

#f1 = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions_ONLY1.csv','r')
#f2 = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions_data1.csv','r')
#data1 = f1.readlines()
#data2 = f2.readlines()
#temp = []
#for eachline1 in data1:
    #split_line1 = eachline1.strip('\n')
    #split_line1 = split_line1.rstrip('"').lstrip('"')

#for eachline2 in data2:
    #split_line2 = eachline2.strip('\n')
    #split_line2 = split_line2.rstrip('"').lstrip('"')
