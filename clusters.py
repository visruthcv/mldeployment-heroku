import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.cluster.hierarchy import fcluster
from sklearn.linear_model import LinearRegression


data = pd.read_excel('input_data.xlsx')

data_for_clust = data.drop(data.columns[0], axis=1).values

dataNorm = preprocessing.scale(data_for_clust)
data_dist = pdist(dataNorm, 'euclidean')
data_linkage = linkage(data_dist, method='average')
last = data_linkage[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
acceleration = np.diff(last, 2)
acceleration_rev = acceleration[::-1]
k = acceleration_rev.argmax() + 2
print(f'clusters: {k}')

km = KMeans(n_clusters=4).fit(dataNorm)

clusters = fcluster(data_linkage, k, criterion='maxclust')

dataK = data
dataK['group_no'] = clusters
writer = pd.ExcelWriter('result.xlsx')
dataK.to_excel(writer, 'KMeans')
writer.save()


def split_by_clusters(data, k):
    while k > 0:
        data_ = data.query('group_no == @k')
        writer = pd.ExcelWriter(f'cluster{k}.xlsx')
        data_.to_excel(writer, f'cluster{k}')
        writer.save()
        k -= 1
    return print('all saved')


split_by_clusters(data, k)

