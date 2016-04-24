f = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions_ONLY.csv','r')
w1 = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions_ONLY1.csv','w')
data = f.readlines()
for eachline in data:
    #print(eachline)
    temp = []
    split_line = eachline.strip('\n')
    split_line = split_line.rstrip('"').lstrip('"')

    temp.append('"' + str(split_line) + '"')
    #print this_row
    w1.write(str(temp).lstrip('[').rstrip(']').replace("'",""))
    #w1.write(str(this_row))
    w1.write("\n")



