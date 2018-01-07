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

house1=pd.read_csv('C:\Users\Dell\Desktop\Project\data\CLEAN_house1.csv', header=None)

house1.columns = ["Timestamp","Aggregate","Fridge","WM","DW","TV","MW","Toaster","HiFi","Kettle","Fan"]

house1['Timestamp'] =(pd.to_datetime(house1['Timestamp'],unit='s')) 

plt.plot(house1['Timestamp'],house1['Aggregate.'])

dataframe2=house1[house1['Timestamp'].astype(str).str.contains('2014-04-04')]

dataframe3=('2014-10-01'<=house1['Timestamp'])&(house1['Timestamp']<='2015-04-30')

housemonthly=house1[dataframe3]

plt.plot(dataframe2['Timestamp'],dataframe2['Aggregate.'])

plt.show()

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
plt.figure()
fig = dataframevis.plot(fontsize = 15, figsize = (15, 6))
plt.tick_params(which=u'major', reset=False, axis = 'y', labelsize = 15)
plt.title('Aggregate power consumption per hour over whole time', fontsize = 16)
plt.ylabel('Watts/hr')
plt.show()


house1.set_index('Timestamp', inplace=True)

ticks = house1.ix[:, ['Aggregate']]

bars = ticks.Aggregate.resample('60min', how='mean')

bars.plot()

dataframehourly=pd.DataFrame({'Timestamp':bars.index, 'Aggregate':bars.values})

dfhourly=dataframehourly.set_index('Timestamp')

bars2=dfhourly.Aggregate.resample('D', how='sum')

dataframedaily=pd.DataFrame({'Timestamp':bars2.index, 'Aggregate':bars2.values})

dfdaily=dataframedaily.set_index('Timestamp')

dfdaily.plot()


def Hourly(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['day'] = df.index.dayofyear
    df['week'] = df.index.weekofyear    
    return df

Electricityhr = Hourly(dfhourly)


from mpl_toolkits.axes_grid1 import make_axes_locatable

ymarks = pd.DataFrame(data = pd.date_range(start = '2014-04-20', end = '2015-04-18', freq = '4W'), columns=['datetime'])
ymarks['date'] = ymarks['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

a1 = ['Mon ', 'Tue ', 'Wed ', 'Thu ', 'Fri ', 'Sat ','Sun ']

a2 = ['12am ', '6am', '12pm', '6pm']

a1 = np.repeat(s1, 4)

a2 = np.tile(s2, 7)

xmarks = np.char.add(a1, a2)

fig = plt.figure(figsize=(20,20))
vis = plt.gca()
ag = vis.agshow(data, vmin =0, vmax = 10000, interpolation='nearest', origin='upper')

d1 = make_axes_locatable(vis)
c1 = d1.append_axes("right", size="5%", pad=0.1)

vis.set_xticks(range(0,140,6))
vis.set_xticklabels(labels = xmarks, fontsize = 10, rotation = 90)

vis.set_yticks(range(0,170,10))
vis.set_yticklabels(labels = ymarks['date'], fontsize = 10)

plt.colorbar(ag, c1=c1)


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
dataframevis2['on/off'] = np.where(dataframevis2['Aggregate']>=5, '1', '0')

dataframevis2.plot()

Electricityhr = Hourly(dfhourly)
Electricityhr['Aggregate-1']=Electricityhr.Aggregate.shift(1)
Electricityhr['Aggregate-2']=Electricityhr.Aggregate.shift(2)
Electricityhr['Aggregate-24']=Electricityhr.Aggregate.shift(24)
Electricityhr['Aggregate-48']=Electricityhr.Aggregate.shift(48)
Electricityhr['day_type'] = np.zeros(len(Electricityhr))
Electricityhr['day_type'][(Electricityhr.index.dayofweek==5)|(Electricityhr.index.dayofweek==6)] = 1
Electricityhr['sin']=np.sin((Electricityhr["hour"].astype(np.float64)*2*pi)/24)
Electricityhr['cos']=np.cos((Electricityhr["hour"].astype(np.float64)*2*pi)/24)
h2=Electricityhr[['Aggregate-1','Aggregate-2','Aggregate-24','Aggregate-48','day_type','sin','cos','Aggregate']]
h2=Electricityhr[['Aggregate-1','Aggregate-2','Aggregate-24','Aggregate-48','day_type','sin','cos','Aggregate']].dropna()

h2.head()