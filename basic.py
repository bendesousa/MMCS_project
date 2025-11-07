import xpress as xp
import pandas as pd
import numpy as np
#xp.init('C:/xpressmp//bin/xpauth.xpr')

#%%
prob = xp.problem(name='basic_bikes')

#Defining the index sets
# number_of_clusters = 250
# number_of_transport_types = 2
# number_of_building_types = 6
# clusters = range(number_of_clusters)
# transport_types = range(number_of_transport_types)
# building_types = range(number_of_building_types)

# transport_names = ['Bicycle', 'Car']
# building_type_names = ['residential', 'commercial', 'school', 
#                        'university', 'hospital', 'library']
# # extracting cluster data
# cluster_data = pd.read_csv('enbraPOIClusterLocations.csv')
# clusters = cluster_data['assigned cluster']

# # extracting trip data
# trip_data = pd.read_csv('all_years_start_to_end.csv')
# # filltering by month and year
# used_month = trip_data[(trip_data['year'] == 2020) & (trip_data['month'] == 1)]

# grouping together the i to j trip numbers
# grouped_trips = (used_month.groupby(['start_station_id', 'end_station_id'])['num_trips'].sum().reset_index()) 

# gathering the unique station ids so they can be converted to i and j
# stations = sorted(trip_data['start_station_id'].unique())
# station_pairs = [(i,j) for i in stations for j in stations]

# #Trips from i to j
# m = {(i,j): 0 for i,j in station_pairs}

# # creating the row values in the m array
# for _, row in grouped_trips.iterrows():
#     m[(int(row['start_station_id']), int(row['end_station_id']))] = int(row['num_trips'])
     
#%%
# extracting trip data
trip_data = pd.read_csv('all_years_start_to_end.csv')
# filltering by month and year
used_month = trip_data[(trip_data['year'] == 2020) & (trip_data['month'] == 1)]
# grouping trips based on start and end station id
m = (
    used_month.groupby(['start_station_id', 'end_station_id'])['num_trips']
    .sum()
    .reset_index()
)

m['start_station_id'] = m['start_station_id'].astype(str).str.strip().astype(int)
m['end_station_id'] = m['end_station_id'].astype(str).str.strip().astype(int)

# mapping station ids to 1-31
unique_stations = sorted(set(m['start_station_id']) | set(m['end_station_id']))
station_map = {old_id: new_id for new_id, old_id in enumerate(unique_stations, start=1)}
m['start_mapped'] = m['start_station_id'].map(station_map)
m['end_mapped'] = m['end_station_id'].map(station_map)

# Creating the matrix
matrix_size = len(unique_stations)
trip_matrix = np.zeros((matrix_size, matrix_size))

# Filling the matrix
for row in m.itertuples(index=False):
    i = (row.start_mapped)-1
    j = (row.end_mapped)-1
    trip_matrix[i,j] = (row.num_trips)
    
trip_frame = pd.DataFrame(
    trip_matrix,
    index=[f"{i}" for i in range(1, matrix_size+1)],
    columns=[f"{j}" for j in range(1, matrix_size+1)])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
# pd.set_option('display.colheader_justify', 'center')

print(trip_frame)

#%%
# # Defining parameters
# # all placeholder values
# manufacture_cost = [400, 200]
# user_cost = 15
# car_users = 0.6
# budget = 1800000
# # need to add data for trip times
# h = {(i,j): 0 for i in clusters for j in clusters}
# # dock maximums
# #M = [1,2,3,4,5,6,7,8,9,10]
# # Building weighting
# p = [1,5,7,10,5,10]
# # environmental proportion
# ep = 0.5
# # car fuel cost
# fuel_cost = 10
 

# # Decision variables
# # Station at cluster i
# x = {i: prob.addVariable(vartype=xp.binary, name='x_{0}'.format(i))
#      for i in clusters}
# # Number of bikes in station at cluster i
# y = {i: prob.addVariable(vartype=xp.integer, name='y_{0}'.format(i))
#      for i in clusters}
# # Number of docks in station at cluster i
# z = {i: prob.addVariable(vartype=xp.integer, name='z_{0}'.format(i))
#      for i in clusters}
# # Number of k buildings in cluster i
# a = {(i,k): prob.addVariable(vartype=xp.integer, name='a_{0}_{1}'.format(i, k))
#      for i in clusters for k in building_types}

# # For concise code
# for i in clusters:
#     bikes_from_i = xp.Sum(m[i,j] for j in clusters)
# for j in clusters:
#     bikes_to_j = xp.Sum(m[i,j] for i in clusters)

# # Total trips
# total_trips = xp.Sum(m[i,j] for i in clusters for j in clusters)

# # Total cost
# total_cost = manufacture_cost[0]*xp.Sum(z[i] for i in clusters) + manufacture_cost[1]*xp.Sum(y[i] for i in clusters)

# #environmental value
# environmental_value = car_users*user_cost*xp.Sum((m[i,j]*h[i,j]*x[i]) for i in clusters for j in clusters)

# # Proportion of demand for a station at cluster i
# demand_share = (1/2*total_trips)*(xp.Sum(bikes_from_i) + xp.Sum(bikes_to_j))

# #%%
# # Constraints
# # station cts
# #prob.addConstraint([z[i] >= x[i] for i in clusters] + [z[i] <= M[i]*x[i] for i in clusters])
# # bike cts
# prob.addConstraint([y[i] >= x[i] for i in clusters] + [y[i] <= z[i] for i in clusters]) 

# # demand cts
# prob.addConstraint(y[i] >= xp.Sum(m[i,j]-m[j,i] for j in clusters))

# # total cost constraint
# prob.addConstraint(total_cost <= budget)

# # environmental constraint
# prob.addConstraint(environmental_value >= ep*fuel_cost)

# # bike movement constraint
# prob.addConstraint(y[i] >= xp.Sum(bikes_from_i-bikes_to_j) for j in clusters)

# # Budget limit exception
# # prob.addConstraint(total_cost <= l[1]*social_value + l[2]*environmental_value)

# #%%
# # Objective function
# social_value = xp.Sum(((p[k]*a[i,k])/(2*total_trips)) for i in clusters for k in building_types)*xp.Sum((bikes_from_i + bikes_to_j)*y[i] for j in clusters)

# prob.setObjective(social_value, sense=xp.maximize)

# prob.write("problem","lp")
# #%%
# xp.setOutputEnabled(True)
# prob.solve()
