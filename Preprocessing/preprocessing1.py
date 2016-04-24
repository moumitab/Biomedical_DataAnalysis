#input the csv file
#go through each line of code and check for NULL
#If NULL present then replace NULL with space or empty string
f = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\Office_20151104.csv','r')
w1 = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\Office_data5.csv','w')
data = f.readlines()
for eachline in data:
    split_line = eachline.strip('\n').split(',')
    newRow = []
    for eachElement in split_line:
        newElement = eachElement.replace('NULL','')
        newRow.append(newElement)
    #this_row = []
    #
    #for element in split_line:
        #this_row.append(element.replace("NULL",""))
    w1.write(str(newRow).lstrip('[').rstrip(']'))
    w1.write('\n')
    #w1.write(str(this_row))
