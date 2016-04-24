f = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions.csv','r')
w1 = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions_data2.csv','w')
data = f.readlines()
for eachline in data:
    #print(eachline)
    this_row = []
    split_line = eachline.strip('\n').split(',')
    for item in split_line:
        this_row.append('"' + str(item) + '"')
    #print this_row
    printEle = str(this_row).lstrip('[').rstrip(']').replace("'","")
    print(printEle)
    w1.write(str(this_row).lstrip('[').rstrip(']').replace("'",""))
    #w1.write(str(this_row))
    w1.write("\n")



