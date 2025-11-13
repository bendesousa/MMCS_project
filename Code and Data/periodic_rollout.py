import xpress as xp
import pandas as pd
import numpy as np
from building_counts import a 
# xp.init('/Applications/FICO Xpress/Xpress Workbench.app/Contents/Resources/xpressmp/bin/xpauth.xpr')

#%%
prob = xp.problem(name='periodic_bikes')

# Defining the index sets
number_of_clusters = 250
number_of_transport_types = 2
number_of_building_types = 6
number_of_periods = 3
clusters = range(number_of_clusters)
transport_types = range(number_of_transport_types)
building_types = range(number_of_building_types)
periods = range(number_of_periods)

transport_names = ['Bicycle', 'Car']
building_type_names = ['residential', 'commercial', 'school', 
                       'university', 'hospital', 'library']

#%%
# Defining parameters
# some placeholder values
manufacture_cost = [500, 40]
hangar = 4000/6
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
# periodic degradation for obj function
degradation_factors = [1, .9, .8]
# periodic % budget available
periodic_pct_budget = [0.5, 0.3, 0.2]
#Max docks for a station
M = 70

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
# Station added period t at cluster i
x_t_i = {(i, t): prob.addVariable(vartype = xp.binary, name='x_{0}_{1}'.format(i, t))
                for i in clusters for t in periods}
# Number of bikes in station added period t at cluster i
y_t_i = {(i, t): prob.addVariable(vartype = xp.integer, name='y_{0}_{1}'.format(i, t))
                for i in clusters for t in periods}
# Number of docks in station at cluster i
z_t_i = {(i, t): prob.addVariable(vartype = xp.integer, name='z_{0}_{1}'.format(i, t))
                for i in clusters for t in periods}
# make certain equations easier on ourselves by summing over periods when necessary
x = {i: sum(x_t_i[i, t] for t in periods) for i in clusters}
y = {i: sum(y_t_i[i, t] for t in periods) for i in clusters}
z = {i: sum(z_t_i[i, t] for t in periods) for i in clusters}

# Trip counts
trips_from_i = m.sum(axis=1)
trips_to_j = m.sum(axis=0)

# For concise code
bikes_from_i = {i: trips_from_i[i] for i in clusters}
bikes_to_j = {j: trips_to_j[j] for j in clusters}

# Total trips
total_trips = m.sum()

# Total cost
total_cost = (manufacture_cost[0]*xp.Sum(z[i] for i in clusters) + manufacture_cost[1]*xp.Sum(y[i] for i in clusters) + hangar*xp.Sum(z[i] for i in clusters))



#environmental value
# environmental_value = car_users*user_cost*xp.Sum((trips_from_i[i]*h*x[i]) for i in clusters)
environmental_value_total = 0
#possible fix for environmental comparison. constraint thoughts below
for i in clusters:
   environmental_value_i = car_users * user_cost * trips_from_i[i] * h * x[i]
   environmental_value_total += environmental_value_i
#proportional environmental values with respect to cost
#for i in clusters:
#    environmental_value_i = car_users * trips_from_i[i] * h * fuel_cost * x[i]

# Minimum coverage cts
building_totals = {}
for i in clusters:
    building_totals[i] = sum(a(i, building_type_names[k]) for k in building_types)

min_coverage = xp.Sum(x[i] * building_totals[i] for i in clusters)

# periodic costs cts
period_costs = {t: 0 for t in periods}
for i in clusters:
    for t in periods:
        period_costs[t] += (manufacture_cost[0]*xp.Sum(z_t_i[i, t]) + manufacture_cost[1]*xp.Sum(y_t_i[i, t]) + hangar * xp.Sum(x_t_i[i, t]))

# Proportion of demand for a station at cluster i
demand_share = {i: 0.5*(bikes_from_i[i] + bikes_to_j[i])/ total_trips
                for i in clusters}

#%%
# Constraints

# demand cts
for i in clusters:
    for t in periods:
        # if station during t at i, then enough bikes to cover demand
        prob.addConstraint(y_t_i[i, t] >= min(max(0, trips_from_i[i] - trips_to_j[i]), M) * x_t_i[i, t])
        # bike cts
        # prob.addConstraint([y_t_i[i, t] >= x_t_i[i, t]]) 
        prob.addConstraint([y_t_i[i, t] <= z_t_i[i, t]])
        # station cts
        prob.addConstraint([z_t_i[i, t] >= x_t_i[i, t]])
        prob.addConstraint([z_t_i[i, t] <= M*x_t_i[i, t]])
        # Slight relaxation of historic station min capacity 
        prob.addConstraint([z_t_i[i, t] >= 10*x_t_i[i, t]])
        # Ensuring empty docks for parking and accounting for unexpected surge in arrivals
        prob.addConstraint([y_t_i[i, t] >= 5*x_t_i[i, t]])
    
# total cost constraint
prob.addConstraint(total_cost <= budget)
# period cost constrain (which should also inherently enforce the total cost constraint but may as well keep both, no?)
for t in periods:
    prob.addConstraint(period_costs[t] <= budget * periodic_pct_budget[t])
# environmental constraint
# prob.addConstraint(environmental_value >= ep*fuel_cost)
prob.addConstraint(environmental_value_total >= ep*fuel_cost)
#prob.addConstraint(environmental_value_i >= ep * fuel_cost * x[i]) proportional to cost
for i in clusters:
    prob.addConstraint(x[i] <= 1)
# Budget limit exception
# prob.addConstraint(total_cost <= l[1]*social_value + l[2]*environmental_value)

# Minimum coverage
# approximately 75% coverage of POIs
prob.addConstraint(min_coverage >= 25000)
#%%
# cluster_weight = {}
# for i in clusters:
#     cluster_weight[i] = sum(
#         (p[k] * a(i, k))
#         for k in building_types
#     )


# I think this will fix our objective function 
cluster_weight = {}
for i in clusters:
    cluster_weight[i] = sum(
        # a_dict is keyed by cluster, building type name, not building type index
        # therefore, a(i, k) for k in building_types would not find a key and return 0
        (p[k] * a(i, building_type_names[k]))
        for k in building_types
    )
    # # this meant that down below, xp.Sum(cluster_weight[i]*(bikes_from_i[i] + bikes_to_j[i])*y[i] 
    #                                                                     for i in clusters)
    #                                                                     = 0*(mij + mji)*yi = 0 for all i in clusters

# Objective function
social_value = xp.Sum(degradation_factors[t]*cluster_weight[i]*(bikes_from_i[i] + bikes_to_j[i])*y_t_i[i, t] 
                                                                        for i in clusters for t in periods)


prob.setObjective(social_value, sense=xp.maximize)

prob.write("periodic_bikes","lp")
#%%
xp.setOutputEnabled(True)
prob.solve()

for t in periods:
    for i in clusters:
        if y_t_i[i, t].getSolution() > 0:
            print(t, i, y_t_i[i, t].getSolution())
        
print("")
        
for t in periods:
    for i in clusters:
        if z_t_i[i, t].getSolution() > 0:
            print(t, i, z_t_i[i, t].getSolution())
print("")    
# for i in clusters:
#     print(f"Cluster {i:3d} | Weight = {cluster_weight[i]*1000:.20f}")

for t in periods:
    print(prob.getSolution(period_costs[t]))    
print(prob.getSolution(total_cost))

# for i in clusters:
#     if environmental_value_i[i]

# print("Total trips:", total_trips)
