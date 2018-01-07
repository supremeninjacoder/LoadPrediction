import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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


trainingdata = pd.DataFrame(data=h2, index=np.arange('2014-04-01 00:00:00', '2014-10-01 00:00:00', dtype='datetime64[h]')).dropna()

testdata = pd.DataFrame(data=h2, index=np.arange('2014-10-02 00:00:00', '2014-12-31 00:00:00', dtype='datetime64[h]')).dropna()

X_train = trainingdata.drop('Aggregate', axis = 1).reset_index().drop('index', axis = 1)

Y_train = trainingdata['Aggregate']

X_test = testdata.drop('Aggregate', axis = 1).reset_index().drop('index', axis = 1)

Y_test = testdata['Aggregate']

linear = LinearRegression()

linear.fit(X_train,Y_train)

y_predict = linear.predict(X_test)

print " Test Accuracy ", linear.score(X_test, Y_test)

print "Coefficients of Linear Regression"

#pd.DataFrame(zip(X_train.columns, y_predict.coef), columns = ['Features', 'Coefficients'])

fig = plt.figure(figsize=(20,10))

plt.legend(loc='lower right')

plt.scatter(X_test.index, Y_test, label='Actual', color='g-')

plt.plot(X_test.index, y_predict, label='Predictions', color='r')

fig = plt.figure(figsize=(10,10))

plt.plot(Y_test, Y_test, c='k')

plt.scatter(Y_test, y_predict, c='r')

