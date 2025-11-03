import xpress as xp
import pandas as pd
import numpy as np
#xp.init('C:/xpressmp//bin/xpauth.xpr')

#%%
prob = xp.problem(name='basic_bikes')

#Defining the index sets
number_of_clusters = 10
number_of_transport_types = 2
clusters = range(number_of_clusters)
transport_types = range(number_of_transport_types)

transport_names = ['Bicycle', 'Car']

#%%
# Defining parameters
# all placeholder values
manufacture_cost = [400, 200]
user_cost = [5,15]
car_users = 0.6
# need to add data for trip times
h = {(i,j): 0 for i in clusters for j in clusters}
# dock maximums
M = [1,2,3,4,5,6,7,8,9,10]

pop_weighting = [0,1,2]
# assigned based on repeating pattern
weight = {i: pop_weighting[idx % len(pop_weighting)]
          for idx, i in enumerate(clusters)}

# Decision variables
# Station at cluster i
x = {i: prob.addVariable(vartype=xp.binary, name='x_{0}'.format(i))
     for i in clusters}
# Number of bikes in station at cluster i
y = {i: prob.addVariable(vartype=xp.integer, name='y_{0}'.format(i))
     for i in clusters}
# Number of docks in station at cluster i
z = {i: prob.addVariable(vartype=xp.integer, name='z_{0}'.format(i))
     for i in clusters}
# Times a bike took a journey i to j
m = {(i,j): prob.addVariable(vartype=xp.integer, name='y_{0}_{1}'.format(i, j))
     for i in clusters for j in clusters}
# Point of interest?
e = {i: prob.addVariable(vartype=xp.binary, name='e_{0}'.format(i))
     for i in clusters}

# For concise code
for i in clusters:
    bikes_from_i = xp.Sum(m[i,j] for j in clusters)
for j in clusters:
    bikes_to_j = xp.Sum(m[i,j] for i in clusters)

# Total trips
total_trips = xp.Sum(m[i,j] for i in clusters for j in clusters)

# Proportion of demand for a station at cluster i
demand_share = (1/2*total_trips)*(xp.Sum(m[i,j] for j in clusters) + xp.Sum(m[j,i] for j in clusters))

# Constraints
# station cts
prob.addConstraint([z[i] >= x[i] for i in clusters] + [z[i] <= M[i]*x[i] for i in clusters])
# bike cts
prob.addConstraint([y[i] >= x[i] for i in clusters] + [y[i] <= z[i] for i in clusters]) 

#bikes_depart = {i: xp.Sum(m[i,j] for j in clusters) for i in clusters}
#bikes_arrive = {j: xp.Sum(m[i,j] for i in clusters) for j in clusters}

# demand cts
prob.addConstraint(y[i] >= xp.Sum(m[i,j]-m[j,i] for j in clusters))

# Constraining stations to be located where there are points of interest
#if weight[i] == e[i] == 0:
    #x[i] = 0
for i in clusters:
    prob.addConstraint(x[i] <= weight[i] + e[i])

#total_cost = (user_cost[1]*xp.Sum(m[i,j]*h[i,j]*x[i] for i in clusters for j in clusters)) - (manufacture_cost[1]*xp.Sum(z[i] for i in clusters)) - (manufacture_cost[2]*xp.Sum(y[i] for i in clusters))

#environmental_value = car_users*user_cost[2]*(xp.Sum(m[i,j]*h[i,j]*x[i] for i in clusters for j in clusters))

#prob.addConstraint(total_cost, environmental_value)
        
# Objective functions 
#social_value = (xp.Sum((weight[i]*e[i]/2*total_trips) for i in clusters))
social_value = xp.Sum(weight[i] * y[i] for i in clusters)
#*(xp.Sum((m[i,j]+m[j,i])*y[i] for j in clusters))

prob.setObjective(social_value, sense=xp.maximize)

prob.write("problem","lp")

xp.setOutputEnabled(True)
prob.solve()
