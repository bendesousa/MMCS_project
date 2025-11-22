import xpress as xp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from building_counts import a 
xp.init('C:/xpressmp//bin/xpauth.xpr')

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
# variable station costs
manufacture_cost = [500, 4000]

# numbers found through extensive research
user_cost = 0.1
car_users = 0.41
budget = 3750000

# average trip time
h = 21.55
# Building weighting
p = [2,2124,55,4602, 2073,2707]

# environmental proportion
ep = 0.000124

# car fuel cost
fuel_cost = 86538603

# periodic degradation for obj function (prioritize better stations first, coverage second)
degradation_factors = [1, .9, .8]
# periodic % budget available
periodic_pct_budget = [0.5, 0.3, 0.2]
#Max hangars for a station
M = 11

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
        
m = cluster_m


#%%
# Decision variables
# Station added period t at cluster i
x_t_i = {(i, t): prob.addVariable(vartype = xp.binary, name='x_{0}_{1}'.format(i, t))
                for i in clusters for t in periods}
# Number of bikes in station added period t at cluster i
y_t_i = {(i, t): prob.addVariable(vartype = xp.integer, name='y_{0}_{1}'.format(i, t))
                for i in clusters for t in periods}
# Number of hangars in station at cluster i
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
total_cost = (manufacture_cost[0]*xp.Sum(y[i] for i in clusters) + manufacture_cost[1]*xp.Sum(z[i] for i in clusters))

#environmental value
environmental_value_total = 0
for i in clusters:
   environmental_value_i = car_users * user_cost * trips_from_i[i] * h * x[i]
   environmental_value_total += environmental_value_i

# Minimum coverage cts
building_totals = {}
for i in clusters:
    building_totals[i] = sum(a(i, building_type_names[k]) for k in building_types)

coverage = xp.Sum(x[i] * building_totals[i] for i in clusters)

# periodic costs cts
period_costs = {t: 0 for t in periods}
for i in clusters:
    for t in periods:
        period_costs[t] += (manufacture_cost[0]*xp.Sum(y_t_i[i, t]) + manufacture_cost[1]*xp.Sum(z_t_i[i, t]))

# Proportion of demand for a station at cluster i
demand_share = {i: 0.5*(bikes_from_i[i] + bikes_to_j[i])/ total_trips
                for i in clusters}

#%%
# Constraints

# demand cts
demand_cts = {}
for i in clusters:
    for t in periods:
        # if station during t at i, then enough bikes to cover demand
        max_bikes = M*6
        demand = max(0, trips_from_i[i] - trips_to_j[i])
        demand_cap = min(min(demand, max_bikes), M*6)
        d = xp.constraint(y_t_i[i, t] >=  demand_cap*x_t_i[i, t])
        prob.addConstraint(d)
        demand_cts[(i,t)] = d
        
        # bike cts
        # open stations have at most 5 bikes per hangar for parking and accounting for unexpected surge in arrivals
        prob.addConstraint([y_t_i[i, t] <= 5*z_t_i[i, t]])
        # station cts
        # open stations have at least one hangar and no more than 11 hangars according to historic system's station capacities
        prob.addConstraint([z_t_i[i, t] >= x_t_i[i, t]])
        prob.addConstraint([z_t_i[i, t] <= M*x_t_i[i, t]])

        # Ensuring stations open with at least 5 bikes
        prob.addConstraint([y_t_i[i, t] >= 5*x_t_i[i, t]])
    
# total cost constraint
cost_cts = xp.constraint(total_cost <= budget)
prob.addConstraint(cost_cts)
# period cost constrain (which should also inherently enforce the total cost constraint but may as well keep both, no?)
periodic_cost_cts = {}
for t in periods:
    c = xp.constraint(period_costs[t] <= budget * periodic_pct_budget[t])
    prob.addConstraint(c)
    periodic_cost_cts[t] = c
    
# environmental constraint
environmental_cts = xp.constraint(environmental_value_total >= ep*fuel_cost)
prob.addConstraint(environmental_cts)

for i in clusters:
    prob.addConstraint(x[i] <= 1)

# Minimum coverage
# approximately 75% coverage of POIs
min_coverage = 25000
coverage_cts = xp.constraint(coverage >= min_coverage)
prob.addConstraint(coverage_cts)

#%%
# I think this will fix our objective function 
cluster_weight = {}
for i in clusters:
    cluster_weight[i] = sum(
        # a_dict is keyed by cluster, building type name, not building type index
        # therefore, a(i, k) for k in building_types would not find a key and return 0
        (p[k] * a(i, building_type_names[k]))
        for k in building_types
    )

# Objective function
social_value = xp.Sum(degradation_factors[t]*cluster_weight[i]*(bikes_from_i[i] + bikes_to_j[i])*y_t_i[i, t] 
                                                                        for i in clusters for t in periods)

prob.setObjective(social_value, sense=xp.maximize)

prob.write("periodic_bikes","lp")

######################Sensitivity analysis ##############################
############Budget#######################
# budget = [1750000, 2250000, 2750000, 3250000, 3750000, 4250000, 4750000, 5250000, 5750000, 6250000, 
#           6750000, 7250000, 7750000, 8250000]
# budget_results = []

# for b in budget:
#     cost_cts.rhs = b
#     for t in periods:
#         periodic_cost_cts[t].rhs = b * periodic_pct_budget[t]
#     xp.setOutputEnabled(False)
#     prob.solve()
#     social_value = prob.attributes.objval
#     coverage_val = prob.getSolution(coverage)
    
#     coverage_per_period = {
#     t: sum(building_totals[i] * x_t_i[i,t].getSolution() for i in clusters)
#     for t in periods
#     }
    
#     budget_results.append({
#         "budget": b,
#         "objective": social_value,
#         "coverage": coverage_val,
#         "deployment": coverage_per_period
#     })

########################Minimum Coverage#######################
# min_coverage = [0, 10000, 15000, 20000, 25000, 30000,]
# coverage_results = []
# for c in min_coverage:
#     coverage_cts.rhs = c
#     xp.setOutputEnabled(False)
#     prob.solve()    
#     stations = sum((sum(x_t_i[i, t].getSolution() for t in periods) 
#                    for i in clusters))
#     bikes = sum((y_t_i[i, t].getSolution())
#                 for i in clusters for t in periods)
#     hangars = sum((z_t_i[i, t].getSolution())
#                   for i in clusters for t in periods)
    
#     social_value = prob.attributes.objval
        
#     coverage_val = sum(
#         building_totals[i] 
#         for i in clusters 
#         if any(x_t_i[i, t].getSolution() > 0.5 for t in periods)
#     )
        
#     coverage_results.append({
#         "min_coverage": c,
#         "objective": social_value,
#         "coverage": coverage_val,
#         "stations": stations,
#         "bikes": bikes,
#         "hangars": hangars
#         # "deployment": coverage_per_period
#     })
######################Sensitivity analysis##############################

#%%
xp.setOutputEnabled(True)
prob.solve()

#%%
# initialize list of cluster-wise results
row_results = []
for i in clusters:
    # index by cluster
    curr_cluster = {'cluster': i}
    for t in periods:
        # get whether opened in t, bikes opened with in t, and hangars opened with in t results for cluster i
        curr_cluster[f"opened during {t}"] = x_t_i[i,t].getSolution()
        curr_cluster[f"bikes added during {t}"] = y_t_i[i,t].getSolution()
        curr_cluster[f"hangars added during {t}"] = z_t_i[i,t].getSolution()
    # add to current cluster results to list of cluster-wise results
    row_results.append(curr_cluster)
# create data frame with cluster-wise results and save as csv
results_frame = pd.DataFrame(row_results)
results_frame.to_csv('periodic_results_by_cluster.csv', index=False)

for t in periods:
    for i in clusters:
        if y_t_i[i, t].getSolution() > 0:
            print(f'Number of bikes in period {t} at cluster {i}: {round(y_t_i[i, t].getSolution())}')
        
print("")
        
for t in periods:
    for i in clusters:
        if z_t_i[i, t].getSolution() > 0:
            print(f'In period {t} at cluster {i} the number of hangars constructed was: {round(z_t_i[i, t].getSolution())}')
print("")    

for t in periods:
    print(f'The cost of operations in period {t} was: £{round(prob.getSolution(period_costs[t]))}')    
print(" ")
print(f'The total cost for the project was: £{round(prob.getSolution(total_cost))}')

print("")

total_stations = sum(prob.getSolution(x[i]) for i in clusters)
print('The total number of stations built was: ', total_stations)
total_buildings = sum(building_totals.values())
print(f'The total number of buildings is {total_buildings}')
print('The total coverage is:', prob.getSolution(coverage))

print("")

#%%
######################Sensitivity analysis ##############################
############Budget#######################
# budget_frame = pd.DataFrame([
#     {
#         "budget": r["budget"],
#         "objective": r["objective"],
#         "coverage": r["coverage"],
#         **{f"period_{t}": r["deployment"][t] for t in periods}
#     }
#     for r in budget_results
# ])
# plt.figure(figsize=(8,6))

# for t in range(3):
#     plt.plot(budget_frame["budget"], budget_frame[f"period_{t}"], label=f"Period {t}")

# plt.plot(
#     budget_frame["budget"],
#     budget_frame["coverage"],
#     marker='D',
#     linestyle='--',
#     linewidth=2,
#     label="Total Coverage"
# )

# plt.xlabel("Budget")
# plt.ylabel("Coverage")
# plt.title("Coverage vs Budget")
# plt.legend()
# plt.grid(True)

# # plt.savefig('budget_vs_periodic_coverage.png', dpi=300, bbox_inches='tight' )
# # plt.savefig('budget_vs_relaxed_periodic_coverage.png', dpi=300, bbox_inches='tight' )
# plt.show()

###################Minimum Coverage########################
# coverage_frame = pd.DataFrame([
#     {
#         "min_coverage": r["min_coverage"],
#         "objective": r["objective"],
#         "coverage": r["coverage"],
#         "stations": r["stations"],
#         "bikes": r["bikes"],
#         "hangars": r["hangars"]
#     }
#     for r in coverage_results
# ])

# plt.figure(figsize=(8,6))
# plt.plot(coverage_frame["min_coverage"], coverage_frame["stations"], marker='o', label="Stations")
# plt.plot(coverage_frame["min_coverage"], coverage_frame["bikes"], marker='s', label="Bikes")
# plt.plot(coverage_frame["min_coverage"], coverage_frame["hangars"], marker='^', label="Hangars")

# plt.xlabel("Minimum Coverage")
# plt.ylabel("Count")
# plt.title("Stations, Bikes, and Hangars vs Minimum Coverage")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('stations,bikes,hangars_vs_min_coverage.png', dpi=300, bbox_inches='tight' )
# # plt.savefig('stations,bikes,hangars_vs_min_coverage(relaxed).png', dpi=300, bbox_inches='tight' )
# plt.show()

# plt.figure(figsize=(8,6))
# plt.plot(coverage_frame["min_coverage"], coverage_frame["coverage"], marker='D', linestyle='--', color='purple', label="Total Coverage")
# plt.xlabel("Minimum Coverage")
# plt.ylabel("Total Coverage")
# plt.title("Total Coverage vs Minimum Coverage")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_coverage_vs_min_coverage.png', dpi=300, bbox_inches='tight' )
# # plt.savefig('total_coverage_vs_min_coverage(relaxed).png', dpi=300, bbox_inches='tight' )
# plt.show()

# plt.figure(figsize=(8,6))
# plt.plot(coverage_frame["min_coverage"], coverage_frame["objective"], marker='x', linestyle='-', color='green', label="Objective Value")
# plt.xlabel("Minimum Coverage")
# plt.ylabel("Objective Value")
# plt.title("Objective Value vs Minimum Coverage")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('objective_vs_min_coverage.png', dpi=300, bbox_inches='tight' )
# # plt.savefig('objective_vs_min_coverage(relaxed).png', dpi=300, bbox_inches='tight' )
# plt.show()