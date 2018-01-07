import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

house1=pd.read_csv('C:\Users\Dell\Desktop\Project\data\CLEAN_house1.csv', header=None)

del house1['Issues']
del house1['Time']
house1['Timestamp'] =(pd.to_datetime(house1['Timestamp'],unit='s')) 
house1.columns = ["Timestamp","Aggregate","Fridge","WM","DW","TV","MW","Toaster","HiFi","Kettle","Fan"]



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

bars.plot()

dataframehourly=pd.DataFrame({'Timestamp':bars.index, 'Aggregate':bars.values})

dfhourly=dataframehourly.set_index('Timestamp')

def Hourly(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['day'] = df.index.dayofyear
    df['week'] = df.index.weekofyear    
    return df


Electricityhr = Hourly(dfhourly)

zeroth=Electricityhr.groupby('hour')['Aggregate'].mean()[0]
first=Electricityhr.groupby('hour')['Aggregate'].mean()[1]
second=Electricityhr.groupby('hour')['Aggregate'].mean()[2]
third=Electricityhr.groupby('hour')['Aggregate'].mean()[3]
fourth=Electricityhr.groupby('hour')['Aggregate'].mean()[4]
fifth=Electricityhr.groupby('hour')['Aggregate'].mean()[5]
sixth=Electricityhr.groupby('hour')['Aggregate'].mean()[6]
seventh=Electricityhr.groupby('hour')['Aggregate'].mean()[7]
eighth=Electricityhr.groupby('hour')['Aggregate'].mean()[8]
ninth=Electricityhr.groupby('hour')['Aggregate'].mean()[9]
tenth=Electricityhr.groupby('hour')['Aggregate'].mean()[10]
eleventh=Electricityhr.groupby('hour')['Aggregate'].mean()[11]
twelveth=Electricityhr.groupby('hour')['Aggregate'].mean()[12]
thirteenth=Electricityhr.groupby('hour')['Aggregate'].mean()[13]
fourteenth=Electricityhr.groupby('hour')['Aggregate'].mean()[14]
fifteenth=Electricityhr.groupby('hour')['Aggregate'].mean()[15]
sixteenth=Electricityhr.groupby('hour')['Aggregate'].mean()[16]
seventeenth=Electricityhr.groupby('hour')['Aggregate'].mean()[17]
eighteenth=Electricityhr.groupby('hour')['Aggregate'].mean()[18]
nineteenth=Electricityhr.groupby('hour')['Aggregate'].mean()[19]
twentieth=Electricityhr.groupby('hour')['Aggregate'].mean()[20]
twentyfirst=Electricityhr.groupby('hour')['Aggregate'].mean()[21]
twentysecond=Electricityhr.groupby('hour')['Aggregate'].mean()[22]
twentythird=Electricityhr.groupby('hour')['Aggregate'].mean()[23]

newhour=[{'dayhour':'0','Aggregate':zeroth},
{'dayhour':'1','Aggregate':first},
{'dayhour':'2','Aggregate':second},
{'dayhour':'3','Aggregate':third},
{'dayhour':'4','Aggregate':fourth},
{'dayhour':'5','Aggregate':fifth},
{'dayhour':'6','Aggregate':sixth},
{'dayhour':'7','Aggregate':seventh},
{'dayhour':'8','Aggregate':eighth},
{'dayhour':'9','Aggregate':ninth},
{'dayhour':'10','Aggregate':tenth},
{'dayhour':'11','Aggregate':eleventh},
{'dayhour':'12','Aggregate':twelveth},
{'dayhour':'13','Aggregate':thirteenth},
{'dayhour':'14','Aggregate':fourteenth},
{'dayhour':'15','Aggregate':fifteenth},
{'dayhour':'16','Aggregate':sixteenth},
{'dayhour':'17','Aggregate':seventeenth},
{'dayhour':'18','Aggregate':eighteenth},
{'dayhour':'19','Aggregate':nineteenth},
{'dayhour':'20','Aggregate':twentieth},
{'dayhour':'21','Aggregate':twentyfirst},
{'dayhour':'22','Aggregate':twentysecond},
{'dayhour':'23','Aggregate':twentythird}]
dataframevis2=pd.DataFrame(newhour)

X1=dataframevis2.iloc[:,0:-1]
y1=dataframevis2.iloc[:,1]


sc_X=StandardScaler()
sc_y=StandardScaler()
X1=sc_X.fit_transform(X1)
y1=sc_y.fit_transform(y1)

regressor = SVR(kernel='rbf')
regressor.fit(X1,y1)

plt.scatter(X1,y1,color='red')
plt.plot(X1,regressor.predict(X1), color='blue')
plt.title('Linear Regression on training set')
plt.xlabel('Hour of day')
plt.ylabel('Power Wh')
plt.show()
print "The test accuracy ", regressor.score(X1, y1)