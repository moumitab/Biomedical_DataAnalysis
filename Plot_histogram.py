import random
import numpy
import matplotlib.pyplot as plt
import plotly.plotly as py
import pandas as pd

file = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\smaller_task\\plot.csv'
data = pd.read_csv(file)
x = data['date']
#dates = plt.dates.date2num(x)
y = data['Time Gap']
print(y)
print(x)

plt.plot_date(x, y)
#pyplot.hist(x, bins, alpha=0.5)
#pyplot.hist(y, bins, alpha=0.5)
#plt.plot(x, y)
plt.xlabel("DateOfVisit")
plt.ylabel("Gap Btw consecutive visit")
plt.title("Date of visit vs Time Gap ")
plt.show()