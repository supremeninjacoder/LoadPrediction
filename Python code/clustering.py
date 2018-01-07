import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib as plt
house1=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house1.csv')
house2=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house2.csv')
house3=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house3.csv')
house4=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house4.csv')
house5=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house5.csv')
house6=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house6.csv')
house7=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house7.csv')
house8=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house8.csv')
house9=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house9.csv')
house10=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house10.csv')
house11=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house11.csv')
house12=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house12.csv')
house13=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house13.csv')
house15=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house15.csv')
house16=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house16.csv')
house17=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house17.csv')
house18=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house18.csv')
house19=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house19.csv')
house20=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house20.csv')
house21=pd.read_csv('C:\Users\Dell\Desktop\Project\housesdataavehourly\house21.csv')

clusterframe=pd.concat([house1,house2,house3,house4,house5,house6,house7,house8,house9,house10,house11,house12,house13,house15,house16,house17,house18,house19,house20,house21], axis=1)
a1=clusterframe[['dayhour','Aggregate']]
X=a1.iloc[:,[0,1]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters =i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters =4, init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=100,c='red',label='Unusually High Consumption ')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=100,c='blue',label='High Aggregate')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=100,c='green',label='Medium Aggregate')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=100,c='cyan',label='Low Aggregate')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Cluster of households')
plt.xlabel('hour of day')
plt.ylabel('Aggregate Consumption')
plt.legend()
plt.show()
