from sklearn.cluster import KMeans

def kmeans(X, df, n_clusters, column):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(X)

    dataCluster = df[['player_id']]
    dataCluster["Cluster"] = kmeans.labels_

    return dataCluster[dataCluster.Cluster == column]