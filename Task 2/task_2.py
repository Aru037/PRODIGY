import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

print(df.head(5))

print(df.info())

print(df.isnull().sum())

X = df.drop(columns=['CustomerID','Gender','Age'], axis=1)

plt.figure(figsize=(15,8))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.show()

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init = 'k-means++' ,random_state=2)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.figure(figsize=(15,8))
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters (K)')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)

kmeans.cluster_centers_

plt.figure(figsize=(15,8))
plt.scatter(X.iloc[Y==0,0], X.iloc[Y==0,1], s=50, c='red', label='Cluster 1')
plt.scatter(X.iloc[Y==1,0], X.iloc[Y==1,1], s=50, c='blue', label='Cluster 2')
plt.scatter(X.iloc[Y==2,0], X.iloc[Y==2,1], s=50, c='green', label='Cluster 3')
plt.scatter(X.iloc[Y==3,0], X.iloc[Y==3,1], s=50, c='yellow', label='Cluster 4')
plt.scatter(X.iloc[Y==4,0], X.iloc[Y==4,1], s=50, c='orange', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', label='C')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()