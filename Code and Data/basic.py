import xpress as xp
import pandas as pd
import numpy as np
from building_counts import a 
import matplotlib.pyplot as plt 
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
# taken from online source
manufacture_cost = [500, 40]
# taken from online source

hangar = 4000

user_cost = 0.1
car_users = 0.41
budget = 3750000
# average of all of the trip times
h = 21.55
# dock maximums
#M = [1,2,3,4,5,6,7,8,9,10]
# Building weighting
p = [2,2124,55,4602, 2073,2707]
# residential = 2
# commercial =(population + number of tourists/4)/number of commercial
# (530,680 + 2,560,000/4)/ 551 = 2124 
# school = (number of 16-18 + admin staff)/number of schools
# 6679 + 3669/ 188 = 55 
# university = 128,869/28 = 4602
# hospital = vistor numbers/ number of hospitals
# 530,680/ 8 / 2 / 16 = 2073
# library = total library memberships / number of libraries
# library = 530,680/4 /49 = 2707

# environmental proportion
ep = 0.000147
# ep = 0.00001
# car fuel cost
fuel_cost = 86538603

#Max docks for a station
M = 66

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

for i in clusters:
    if x[i] == 1:
        total_cost += (hangar/6*z[i])

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


# Proportion of demand for a station at cluster i
demand_share = {i: 0.5*(bikes_from_i[i] + bikes_to_j[i])/ total_trips
                for i in clusters}

#%%
# Constraints

# demand cts
for i in clusters:
    # this means that any time there are more trips from i than to i, we are forced to open a station
    # even if the station only covered ~50 POIs, we are forcing it to open
    prob.addConstraint(y[i] >= min(max(0, trips_from_i[i] - trips_to_j[i]), M) * x[i])
    # bike cts
    prob.addConstraint([y[i] >= x[i]]) 
    prob.addConstraint([y[i] <= 5*z[i]])
    # station cts
    prob.addConstraint([z[i] >= x[i]])
    prob.addConstraint([z[i] <= M*x[i]])
    # Slight relaxation of historic station min capacity 
    # prob.addConstraint([z[i] >= x[i]])
    # Ensuring empty docks for parking and accounting for unexpected surge in arrivals
    prob.addConstraint([y[i] >= 5*x[i]])
 
    
# total cost constraint
###### Binding constraint #########
cost_cts = xp.constraint(total_cost <= budget)
prob.addConstraint(cost_cts)
 
# environmental constraint
###### Binding constraint #########
prob.addConstraint(environmental_value_total >= ep*fuel_cost)
#prob.addConstraint(environmental_value_i >= ep * fuel_cost * x[i]) proportional to cost

# Budget limit exception
# prob.addConstraint(total_cost <= l[1]*social_value + l[2]*environmental_value)

# Minimum coverage
# approximately 75% coverage of POIs
###### Binding constraint #########
# coverage_cts = xp.constraint(min_coverage >= 25000)
# prob.addConstraint(coverage_cts)
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
social_value = xp.Sum(cluster_weight[i]*(bikes_from_i[i] + bikes_to_j[i])*y[i] 
                                                                        for i in clusters)


prob.setObjective(social_value, sense=xp.maximize)

prob.write("problem","lp")

######################Sensitivity analysis ##############################
############Budget 
poss_budgets = [1750000, 2250000, 2750000, 3250000, 3750000, 4250000, 4750000, 5250000, 5750000, 6250000, 6750000, 7250000, 7750000, 8250000 ]
results = []

for b in poss_budgets:
    cost_cts.rhs = b
    xp.setOutputEnabled(True)
    prob.solve()
    social_value = prob.getObjVal()
    coverage_val = prob.getSolution(min_coverage)
    
    print(f'Budget {b}: Objective = {social_value}, Cluster coverage = {coverage_val}')
    
    results.append({
        "budget": b,
        "objective": social_value,
        "coverage": coverage_val
    })
    
######################Sensitivity analysis##############################
#%%
# xp.setOutputEnabled(True)
# prob.solve()

for i in clusters:
    if y[i].getSolution() > 0:
        print("The number of bikes placed at cluster", i, ": ", y[i].getSolution())
        
print("")
        
for i in clusters:
    if z[i].getSolution() > 0:
        print("The number of docks opened at cluster ", i, ": ", z[i].getSolution())
        
print("")
    
# for i in clusters:
#     print(f"Cluster {i:3d} | Weight = {cluster_weight[i]*1000:.20f}")
    
print("The total cost of this project is : £", prob.getSolution(total_cost))

print("")

total_stations = sum(prob.getSolution(x[i]) for i in clusters)
print("The total number of stations opened is:", total_stations)

print("")

environmental_value_value = sum(
    car_users * user_cost * trips_from_i[i] * h * prob.getSolution(x[i])
    for i in clusters
)
print("The total environmental value is:", environmental_value_value)

print("")

print("The coverage of the the POIs is:", prob.getSolution(min_coverage))

# print("Total trips:", total_trips)
df = pd.DataFrame(results)
plt.figure(figsize=(10,5))
# plt.plot(df["budget"], df["objective"], marker="o", label="Objective")
plt.plot(df["budget"], df["coverage"], marker="x", label="Cluster coverage")
plt.xlabel("Budget")
plt.ylabel("Value")
plt.title("Sensitivity of Objective and Cluster Coverage to Budget")
plt.legend()
plt.grid(True)
plt.show()
