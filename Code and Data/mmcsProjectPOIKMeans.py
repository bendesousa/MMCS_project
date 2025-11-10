import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
"""
we are going to implement k-means clustering to take whatever fuckload of geographic points you want and put them into k groups
according to how close together they are.

we arbitrarily decide k = 250 = number of potential stations to build. that just kinda seems like a nice number of candidates to consider. that can change quite easily

i had a vba script that did the clustering once, but apparently the uni won't let me use my account to edit in excel on a mac, so here we are with python.
honestly though, kinda great because i can actually just use pre-existing libraries to do it instead of writing four different functions and
doing it all by hand. also we want to do a lot more with it so yeah
"""
# create a data frame with the data from edinburgh pois
pois = pd.read_csv("edinburgh_pois.csv")

category_weights = {
'library':10,
'school':7,
'university':7,
'residential':1,
'commercial':5,
'hospital':5
}

# Map category names to weights (default to 1 if not found)
pois['category_weight'] = pois['category'].map(category_weights).fillna(1).astype(int)
pois_weighted = pois.loc[pois.index.repeat(pois['category_weight'])].reset_index(drop=True)

# define an nx2 vector with the latitude and longitude of each poi
X = pois_weighted[['lat', 'lon']].values
# arbitrarily we are considering k potential stations sited at the geographic mean of the pois they serve
k = 250
# thank you sci-kit for writing the hard part for me
kmeans = KMeans(n_clusters=k, random_state=42)
# add each poi's assigned cluster to the data frame
pois_weighted['assigned cluster'] = kmeans.fit_predict(X)

cluster_assignments = (
    pois_weighted.groupby(['lat', 'lon'])['assigned cluster']
    .agg(lambda x: x.mode()[0])
    .reset_index()
)
pois = pois.merge(cluster_assignments, on=['lat', 'lon'], how='left')
"""
plt.scatter(pois['lon'], pois['lat'], c=pois['cluster'], cmap='tab10')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-Means Clustering of Points of Interest')
plt.show()
"""

# find the geographic mean of each cluster and create a new data frame with it
cluster_centers = (
    pois.groupby('assigned cluster')[['lat', 'lon']]
    .mean()
    .reset_index()
    .rename(columns={'assigned cluster': 'cluster'})
)

# if you want cool matplots of each POI and its cluster and its cluster's geographic mean put this back in

plt.scatter(pois['lon'], pois['lat'], c=pois['assigned cluster'], cmap='tab10', alpha=0.5)
plt.scatter(cluster_centers['lon'], cluster_centers['lat'], c='black', marker='x', s=25, label='Cluster Centers')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('K-Means Clustering of Points of Interest')
# plt.show()
plt.savefig('trans 250 Means Clustering of edinburgh_pois.png', transparent = True)
plt.savefig('250 Means Clustering of edinburgh_pois.png')


radius = 700 / 111000  # radius in degrees (every 1 degree is approximately 111,000 meters)

# Create a new column to store clusters within range
pois['nearby_clusters'] = [[] for _ in range(len(pois))]

# Compute for each cluster center
for _, center in cluster_centers.iterrows():
    cluster_id = center['cluster']
    # Compute simple Euclidean distance in degrees
    dist = np.sqrt((pois['lat'] - center['lat'])**2 + (pois['lon'] - center['lon'])**2)
    # Add this cluster number to each POI that falls within radius
    pois.loc[dist <= radius, 'nearby_clusters'] = pois.loc[dist <= radius, 'nearby_clusters'].apply(
        lambda lst: lst + [cluster_id]
    )

# convert np floats to ints for cleaner reading
pois['nearby_clusters'] = pois['nearby_clusters'].apply(lambda lst: [int(x) for x in lst])

# Turn the list of nearby clusters into separate rows
exploded = pois.explode('nearby_clusters')

# Count combinations of cluster + category
counts = exploded.groupby(['nearby_clusters', 'category']).size().unstack(fill_value=0)

# Rename the index
counts.index.name = 'cluster'
counts = counts.reset_index()

# Merge back with cluster_centers
cluster_centers = cluster_centers.merge(counts, on='cluster', how='left').fillna(0)

# write our two dataframes to csv files. the first has each poi and its assigned cluster. the second has each cluster and its geographical mean
pois.to_csv("enbraPOIClusterAssingments.csv", index = False)
cluster_centers.to_csv("enbraPOIClusterLocations.csv", index=False)
