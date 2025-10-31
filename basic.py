import xpress as xp
import pandas as pd
import numpy as np
xpress.init('C:/xpressmp//bin/xpauth.xpr')

prob = xp.problem(name='basic_bikes')

#Defining the index sets
number_of_clusters = 10
number_of_transport_types = 2
clusters = range(number_of_clusters)
transport_types = range(number_of_transport_types)

transport_names = ['Bicycle', 'Car']

# Defining parameters
# all placeholder values
manufacture_cost = [400, 200]
user_cost = [5,15]
car_users = 0.6
# need to add data for trip times
h = {(i,j): 0 for i in clusters for j in clusters}

pop_weighting = [0,1,2]
# assigned based on repeating pattern
weight = {i: pop_weighting[idx % len(pop_weighting)]
          for idx, i in enumerate(clusters)}

# Decision variables
# Station at cluster i
x = {(i): prob.addVariable(vartype=xp.binary, name='x{0}'.format(i))
     for i in clusters}
# Number of bikes in station at cluster i
y = {(i): prob.addVariable(vartype=xp.integer, name='y{0}'.format(i))
     for i in clusters}
# Number of docks in station at cluster i
z = {(i): prob.addVariable(vartype=xp.integer, name='z{0}'.format(i))
     for i in clusters}
# Times a bike took a journey i to j
m = {(i,j): prob.addVariable(vartype=xp.integer, name='y{0}_{1}'.format(i, j))
     for i in clusters for j in clusters}
# Point of interest?
e = {(i): prob.addVariable(vartype=xp.binary, name='e{0}'.format(i))
     for i in clusters}

# For concise code
for i in clusters:
    bikes_from_i = xp.sum(m[i,j] for j in clusters)
for j in clusters:
    bikes_to_j = xp.sum(m[i,j] for i in clusters)

# Total trips
total_trips = xp.sum(m[i,j] for i in clusters for j in clusters)

# Proportion of demand for a station at cluster i
demand_share = (1/2*total_trips)(xp.sum(m[i,j] for j in clusters) + xp.sum(m[j,i] for j in clusters))

# Constraints
station_cts = [x[i] <= z[i] <= M[i]*x[i]]
bike_cts = [x[i] <= y[i] <= z[i]]

if x[i] == 1:
    demand_cts = y[i] >= xp.sum(m[i,j]-m[j,i] for j in clusters)

# Constraining stations to be located where there are points of interest
if weight[i] == e[i] == 0:
    x[i] = 0
for i in clusters:
    x[i] <= weight[i] + e[i]

prob.addConstraint(station_cts, bike_cts)

# Objective functions 
social_value = (xp.sum((weight[i]*e[i]/2*total_trips) for i in clusters))*(xp.sum((m[i,j]+m[j,i])*y[i] for j in clusters))

total_cost = (user_cost[1]*xp.sum(m[i,j]*h[i,j]*x[i] for i in clusters for j in clusters)) - (manufacture_cost[1]*xp.sum(z[i] for i in clusters)) - (manufacture_cost[2]*xp.sum(y[i] for i in clusters))

environmental_value = car_users*user_cost[2]*(xp.sum(m[i,j]*h[i,j]*x[i] for i in clusters for j in clusters))

prob.write("problem","lp")

xp.setOutputEnabled(True)
prob.solve()
