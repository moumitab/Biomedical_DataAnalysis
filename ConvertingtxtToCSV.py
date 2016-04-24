#Code to convert a .txt file into .csv
import csv

#hospital records
txt_file = r"C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Lab_20151104_sorted.txt"
csv_file = r"C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Lab_20151104_sorted.csv"

#office records
#txt_file = r"C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Office_20151104_sorted.txt"
#csv_file = r"C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Office_20151104_sorted.csv"

# use 'with' if the program isn't going to immediately terminate
# so you don't leave files open
# the 'b' is necessary on Windows
# it prevents \x1a, Ctrl-z, from ending the stream prematurely
# and also stops Python converting to / from different line terminators
# On other platforms, it has no effect
in_txt = csv.reader(open(txt_file, "rb"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'wb'))

out_csv.writerows(in_txt)