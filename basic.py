import xpress as xp
import pandas as pd
import numpy as np
from building_counts import a 
#xp.init('C:/xpressmp//bin/xpauth.xpr')

#%%
prob = xp.problem(name='basic_bikes')

# Defining the index sets
number_of_clusters = 250
number_of_transport_types = 2
number_of_building_types = 6
clusters = range(number_of_clusters)
transport_types = range(number_of_transport_types)
building_types = range(number_of_building_types)

transport_names = ['Bicycle', 'Car']
building_type_names = ['residential', 'commercial', 'school', 
                       'university', 'hospital', 'library']

#%%
# Defining parameters
# some placeholder values
manufacture_cost = [20, 1]
user_cost = 15
car_users = 0.6
budget = 1800000
# need to add data for trip times
h = 21.55
# dock maximums
#M = [1,2,3,4,5,6,7,8,9,10]
# Building weighting
p = [1,5,7,10,5,10]
# environmental proportion
ep = 0.5
# car fuel cost
fuel_cost = 10

#Max docks for a station
M = 20

# Trips from cluster i to j
m = np.load('m_matrix.npy')

# Assigning trips to clusters
station_assignments = pd.read_csv('stations_assigned.csv')
assigned_stations = station_assignments[station_assignments['assigned_cluster'] != -1]
station_to_cluster = dict(zip(assigned_stations['station_id'], assigned_stations['assigned_cluster']))

station_ids_in_matrix = sorted(assigned_stations['station_id'].tolist())
station_index_in_matrix = {station_id: idx for idx, station_id in enumerate(station_ids_in_matrix)}

n = number_of_clusters
cluster_m = np.zeros((n,n))

for i_station, i_cluster in station_to_cluster.items():
    for j_station, j_cluster in station_to_cluster.items():
        i_idx = i_cluster
        j_idx = j_cluster
        i_orig = station_index_in_matrix[i_station]
        j_orig = station_index_in_matrix[j_station]
        cluster_m[i_idx, j_idx] = m[i_orig, j_orig]
        
# cluster_m[cluster_m==0] = m[cluster_m==0]

m = cluster_m


#%%
# Decision variables
# Station at cluster i
x = {i: prob.addVariable(vartype=xp.binary, name='x_{0}'.format(i))
     for i in clusters}
# Number of bikes in station at cluster i
y = {i: prob.addVariable(vartype=xp.integer, lb=0, name='y_{0}'.format(i))
     for i in clusters}
# Number of docks in station at cluster i
z = {i: prob.addVariable(vartype=xp.integer, lb=0, name='z_{0}'.format(i))
     for i in clusters}

# Trip counts
trips_from_i = m.sum(axis=1)
trips_to_j = m.sum(axis=0)

# For concise code
bikes_from_i = {i: trips_from_i[i] for i in clusters}
bikes_to_j = {j: trips_to_j[j] for j in clusters}

# Total trips
total_trips = m.sum()

# Total cost
total_cost = manufacture_cost[0]*xp.Sum(z[i] for i in clusters) + manufacture_cost[1]*xp.Sum(y[i] for i in clusters)

#environmental value
environmental_value = car_users*user_cost*xp.Sum((trips_from_i[i]*h*x[i]) for i in clusters)

# Proportion of demand for a station at cluster i
demand_share = {i: 0.5*(bikes_from_i[i] + bikes_to_j[i])/ total_trips
                for i in clusters}

#%%
# Constraints

# demand cts
for i in clusters:
    prob.addConstraint(y[i] >= min(max(0, trips_from_i[i] - trips_to_j[i]), M))
    # bike cts
    prob.addConstraint([y[i] >= x[i]]) 
    prob.addConstraint([y[i] <= z[i]])
    # station cts
    prob.addConstraint([z[i] >= x[i]])
    prob.addConstraint([z[i] <= M*x[i]])
    
# total cost constraint
prob.addConstraint(total_cost <= budget)

# environmental constraint
prob.addConstraint(environmental_value >= ep*fuel_cost)

# Budget limit exception
# prob.addConstraint(total_cost <= l[1]*social_value + l[2]*environmental_value)

#%%
cluster_weight = {}
for i in clusters:
    cluster_weight[i] = sum(
        (p[k] * a(i, k)) / (2 * total_trips)
        for k in building_types
    )

# Objective function
social_value = xp.Sum(cluster_weight[i]*(bikes_from_i[i] + bikes_to_j[i])*y[i] 
                                                                        for i in clusters)


prob.setObjective(social_value, sense=xp.maximize)

prob.write("problem","lp")
#%%
xp.setOutputEnabled(True)
prob.solve()
