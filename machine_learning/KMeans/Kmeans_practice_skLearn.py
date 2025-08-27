import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('./my_CSV_files/student_clustering.csv')

df.head()

df.shape

plt.scatter(df['cgpa'], df['iq'])

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)


plt.plot(range(1,11), wcss)

X = df.iloc[:,:].values
km = KMeans(n_clusters=4)
y_mean = km.fit_predict(X)
y_mean


X[y_mean == 2]


plt.scatter(X[y_mean  == 0,0],X[y_mean  == 0,1],color='blue')
plt.scatter(X[y_mean  == 1,0],X[y_mean  == 1,1],color='red')
plt.scatter(X[y_mean  == 2,0],X[y_mean  == 2,1],color='green')
plt.scatter(X[y_mean  == 3,0],X[y_mean  == 3,1],color='yellow')