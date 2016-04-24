f = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions.csv','r')
w1 = open('C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\conditions_data.csv','w')
data = f.readlines()
for eachline in data:
    print(eachline)
    split_line = eachline.strip('\n').split(',')
    this_row = ['"' + str(split_line) + '"' for item in split_line]
    print (this_row)
    w1.write(str(this_row))



