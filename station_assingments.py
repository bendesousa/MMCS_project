import pandas as pd
import numpy as np


cluster_centers = pd.read_csv('enbraPOIClusterLocations.csv')
station_locations = pd.read_csv('station_data.csv')
station_ids = pd.read_csv('jan_stations.csv')
station_ids_needed = station_ids['required'].tolist()

required_stations = station_locations[station_locations['station_id'].isin(station_ids_needed)]

station_locations['assigned_cluster'] = -1
assigned_clusters = set()


# Compute for each cluster center
for idx, station in station_locations.iterrows():
    # Compute simple Euclidean distance in degrees
    dist = np.sqrt((cluster_centers['lat'] - station['lat'])**2 + (cluster_centers['lon'] - station['lon'])**2)
    # sort clusters by distance
    sorted_clusters = cluster_centers.iloc[dist.argsort()]['cluster']
    for cluster_id in sorted_clusters:
        if cluster_id not in assigned_clusters:
            station_locations.at[idx, 'assigned_cluster'] = cluster_id
            assigned_clusters.add(cluster_id)
            break
        
station_locations.to_csv('stations_assigned.csv', index=False)
