
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import numpy.linalg as lin 
import matplotlib
import sklearn.metrics
from sklearn import gaussian_process
from sklearn import cross_validation
import sklearn.decomposition

pd.options.display.mpl_style = 'default'

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from tabulate import tabulate
import datetime
import time
from StringIO import StringIO
import numpy.linalg as lin # module for performing linear algebra operations
import matplotlib
from pandas import ExcelWriter

house1=pd.read_csv('C:\Users\Dell\Desktop\New folder\Project\data\CLEAN_house20.csv')

del house1['Issues']
del house1['Time']

house1.columns = ["Timestamp","Aggregate","Fridge","WM","DW","TV","MW","Toaster","HiFi","Kettle","Fan"]

house1['Timestamp'] =(pd.to_datetime(house1['Timestamp'],unit='s')) 


def hr_func(ts):
    return ts.hour

house1['time_hour'] = house1['Timestamp'].apply(hr_func)
zerohour=house1.loc[house1['time_hour'] == 0, 'Aggregate'].mean()
firsthour=house1.loc[house1['time_hour'] == 1, 'Aggregate'].mean()
secondhour=house1.loc[house1['time_hour'] == 2, 'Aggregate'].mean()
thirdhour=house1.loc[house1['time_hour'] == 3, 'Aggregate'].mean()
fourhour=house1.loc[house1['time_hour'] == 4, 'Aggregate'].mean()
fivehour=house1.loc[house1['time_hour'] == 5, 'Aggregate'].mean()
sixthour=house1.loc[house1['time_hour'] == 6, 'Aggregate'].mean()
sevenhour=house1.loc[house1['time_hour'] == 7, 'Aggregate'].mean()
eighthour=house1.loc[house1['time_hour'] == 8, 'Aggregate'].mean()
ninehour=house1.loc[house1['time_hour'] == 9, 'Aggregate'].mean()
tenhour=house1.loc[house1['time_hour'] == 10, 'Aggregate'].mean()
elevenhour=house1.loc[house1['time_hour'] == 11, 'Aggregate'].mean()
twelvehour=house1.loc[house1['time_hour'] == 12, 'Aggregate'].mean()
thirteenhour=house1.loc[house1['time_hour'] == 13, 'Aggregate'].mean()
fourteenhour=house1.loc[house1['time_hour'] == 14, 'Aggregate'].mean()
fifteenhour=house1.loc[house1['time_hour'] == 15, 'Aggregate'].mean()
sixteenhour=house1.loc[house1['time_hour'] == 16, 'Aggregate'].mean()
seventeenhour=house1.loc[house1['time_hour'] == 17, 'Aggregate'].mean()
eighteenhour=house1.loc[house1['time_hour'] == 18, 'Aggregate'].mean()
nineteenhour=house1.loc[house1['time_hour'] == 19, 'Aggregate'].mean()
twentyhour=house1.loc[house1['time_hour'] == 20, 'Aggregate'].mean()
twentyonehour=house1.loc[house1['time_hour'] == 21, 'Aggregate'].mean()
twentytwohour=house1.loc[house1['time_hour'] == 22, 'Aggregate'].mean()
twentythreehour=house1.loc[house1['time_hour']==23,'Aggregate'].mean()

#Aggregate., Aggregate,etc hourly plot for a month, yearly plot , daily plot, day of week/holiday etc.,interval of day plot, season plot, occupants diff. houses consumption
hour=[{'dayhour':'0','Aggregate':zerohour},
{'dayhour':'1','Aggregate':firsthour},
{'dayhour':'2','Aggregate':secondhour},
{'dayhour':'3','Aggregate':thirdhour},
{'dayhour':'4','Aggregate':fourhour},
{'dayhour':'5','Aggregate':fivehour},
{'dayhour':'6','Aggregate':sixthour},
{'dayhour':'7','Aggregate':sevenhour},
{'dayhour':'8','Aggregate':eighthour},
{'dayhour':'9','Aggregate':ninehour},
{'dayhour':'10','Aggregate':tenhour},
{'dayhour':'11','Aggregate':elevenhour},
{'dayhour':'12','Aggregate':twelvehour},
{'dayhour':'13','Aggregate':thirteenhour},
{'dayhour':'14','Aggregate':fourteenhour},
{'dayhour':'15','Aggregate':fifteenhour},
{'dayhour':'16','Aggregate':sixteenhour},
{'dayhour':'17','Aggregate':seventeenhour},
{'dayhour':'18','Aggregate':eighteenhour},
{'dayhour':'19','Aggregate':nineteenhour},
{'dayhour':'20','Aggregate':twentyhour},
{'dayhour':'21','Aggregate':twentyonehour},
{'dayhour':'22','Aggregate':twentytwohour},
{'dayhour':'23','Aggregate':twentythreehour}]

dataframevis=pd.DataFrame(hour)

house1.set_index('Timestamp', inplace=True)

ticks = house1.ix[:, ['Aggregate']]

bars = ticks.Aggregate.resample('60min', how='mean')

#bars.plot()

dataframehourly=pd.DataFrame({'Timestamp':bars.index, 'Aggregate':bars.values})

dfhourly=dataframehourly.set_index('Timestamp')



def addHourlyTimeFeatures(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['day'] = df.index.dayofyear
    df['week'] = df.index.weekofyear    
    return df

hourlyElectricity = addHourlyTimeFeatures(dfhourly)

hourlyElectricity['Aggregate-1']=hourlyElectricity.Aggregate.shift(1)

hourlyElectricity['Aggregate-2']=hourlyElectricity.Aggregate.shift(2)

hourlyElectricity['Aggregate-24']=hourlyElectricity.Aggregate.shift(24)

hourlyElectricity['Aggregate-48']=hourlyElectricity.Aggregate.shift(48)

hourlyElectricity['day_type'] = np.zeros(len(hourlyElectricity))

hourlyElectricity['day_type'][(hourlyElectricity.index.dayofweek==5)|(hourlyElectricity.index.dayofweek==6)] = 1

hourlyElectricity['sin']=np.sin((hourlyElectricity["hour"].astype(np.float64)*2*3.14)/24)

hourlyElectricity['cos']=np.cos((hourlyElectricity["hour"].astype(np.float64)*2*3.14)/24)

h2=hourlyElectricity[['Aggregate-1','Aggregate-2','Aggregate-24','Aggregate-48','day_type','sin','cos','Aggregate']]

h2=hourlyElectricity[['Aggregate-1','Aggregate-2','Aggregate-24','Aggregate-48','day_type','sin','cos','Aggregate']].dropna()

trainingdata = h2['2014-04-01 00:00:00':'2014-04-20 00:00:00']

testdata = h2['2014-04-21 00:00:00':'2014-04-27 00:00:00']

X_train = trainingdata.values[:,0:-1]

Y_train = trainingdata.values[:,7]

X_test = testdata.values[:,0:-1]

Y_test = testdata.values[:,7]


def predictAll(theta, nugget, trainX, trainY, testX, testY, testSet, title):

    gp = gaussian_process.GaussianProcess(theta0=theta, nugget =nugget)
    gp.fit(trainX, trainY)

    predictedY, MSE = gp.predict(testX, eval_MSE = True)
    sigma = np.sqrt(MSE)

    results = testSet.copy()
    results['predictedY'] = predictedY
    results['sigma'] = sigma

    print "Train score R2:", gp.score(trainX, trainY)
    print "Test score R2:", sklearn.metrics.r2_score(testY, predictedY)

    plt.figure(figsize = (9,8))
    plt.scatter(testY, predictedY)
    plt.plot([min(testY), max(testY)], [min(testY), max(testY)], 'r')
    plt.xlim([min(testY), max(testY)])
    plt.ylim([min(testY), max(testY)])
    plt.title('Predicted vs. observed: ' + title)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.show()
    
    return gp, results

gp_dailyElectricity, results_dailyElectricity = predictAll(0.75, 0.56, X_train, Y_train, 
                                  X_test, Y_test, testdata, 'Daily Electricity')

def plotGP(testY, predictedY, sigma):
    fig = plt.figure(figsize = (20,6))
    plt.plot(testY.values, 'r.', markersize=10, label=u'Observations')
    plt.plot(predictedY.values, 'b-', label=u'Prediction')
    x = range(len(testY))
    plt.fill(np.concatenate([x, x[::-1]]), np.concatenate([predictedY - 1.9600 * sigma, (predictedY + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')

    
subset = results_dailyElectricity['2014-04-21 00:00:00':'2014-04-27 00:00:00']
testY = subset['Aggregate']
predictedY = subset['predictedY']
sigma = subset['sigma']

plotGP(testY, predictedY, sigma)

plt.ylabel('Electricity (kWh)', fontsize = 13)
plt.title('Gaussian Process Regression: Daily Electricity Prediction', fontsize = 17)
plt.legend(loc='upper right')
plt.xlim([0, len(testY)])
plt.ylim([0,1000])

xTickLabels = pd.DataFrame(data = subset.index[np.arange(0,len(subset.index),10)], columns=['datetime'])
xTickLabels['date'] = xTickLabels['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
ax = plt.gca()
ax.set_xticks(np.arange(0, len(subset), 10))
ax.set_xticklabels(labels = xTickLabels['date'], fontsize = 13, rotation = 90)
plt.show()