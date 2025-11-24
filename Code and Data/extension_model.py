import xpress as xp
import pandas as pd
import numpy as np
from building_counts import a 
# xp.init('C:/xpressmp//bin/xpauth.xpr')

#%%
def period_prob(previous_results, period, m = np.load('m_matrix.npy')):
	
	prob = xp.problem(name='periodic_bikes')

	if "cluster" not in previous_results.columns:
		raise ValueError("previous_results must include a 'cluster' column.")

	prev = previous_results.copy()
	prev = prev.sort_values("cluster").reset_index(drop=True)   # deterministic row order
	prev = prev.set_index("cluster")

	# check our periods are working right
	if period < 0:
		raise ValueError("period must be >= 0")
	number_of_periods = period + 1
	periods = range(number_of_periods)

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
	# variable station costs
	manufacture_cost = [500, 4000]

	# numbers found through extensive research
	user_cost = 0.1
	car_users = 0.41
	budget = 3750000
	# average trip time
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
	ep = 0.000124

	# car fuel cost
	fuel_cost = 86538603

	# periodic degradation for obj function (prioritize better stations first, coverage second)
	degradation_factors = [1, .9, .8]
	# periodic % budget available
	periodic_pct_budget = [0.5, 0.3, 0.2]
	periodic_pct_sec_obj = [1/3, 1/3, 1/3]
	#Max hangars for a station
	M = 11

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
	environmental_value_total = xp.Sum((car_users*user_cost*trips_from_i[i]*h*x[i]) for i in clusters)
	# environmental_value_total = 0
	# #possible fix for environmental comparison. constraint thoughts below
	# for i in clusters:
	# 	environmental_value_i = car_users * user_cost * trips_from_i[i] * h * x[i]
	# 	environmental_value_total += environmental_value_i
	#proportional environmental values with respect to cost
	#for i in clusters:
	#    environmental_value_i = car_users * trips_from_i[i] * h * fuel_cost * x[i]

	# Minimum coverage cts
	building_totals = {}
	for i in clusters:
		building_totals[i] = sum(a(i, building_type_names[k]) for k in building_types)

	min_coverage = xp.Sum(x[i] * building_totals[i] for i in clusters)

	# periodic costs cts
	period_costs = {t: (manufacture_cost[0]*xp.Sum(y_t_i[i, t] for i in clusters) + manufacture_cost[1]*xp.Sum(z_t_i[i, t] for i in clusters)) for t in periods}

	# Proportion of demand for a station at cluster i
	demand_share = {i: 0.5*(bikes_from_i[i] + bikes_to_j[i])/ total_trips
					for i in clusters}

	#%%
	# Constraints

	# demand cts
	for i in clusters:
		for t in periods:
			if t == period:
				# if station during t at i, then enough bikes to cover demand
				prob.addConstraint(y_t_i[i, t] >= min(max(0, trips_from_i[i] - trips_to_j[i]), M*6) * x_t_i[i, t])
				# bike cts
				# open stations have at most 5 bikes per hangar for parking and accounting for unexpected surge in arrivals
				prob.addConstraint([y_t_i[i, t] <= 5*z_t_i[i, t]])
				# station cts
				# open stations have at least one hangar and no more than 11 hangars according to historic system's station capacities
				prob.addConstraint([z_t_i[i, t] >= x_t_i[i, t]])
				prob.addConstraint([z_t_i[i, t] <= M*x_t_i[i, t]])

				# Ensuring stations open with at least 5 bikes
				prob.addConstraint([y_t_i[i, t] >= 5*x_t_i[i, t]])

		# for t in periods:
			if t < period:

				prev_open_col = f"opened during {t}"
				prev_bikes_col = f"bikes added during {t}"
				prev_hangars_col = f"hangars added during {t}"

				if prev_open_col in prev.columns:
					val = int(prev.at[i, prev_open_col]) if not pd.isna(prev.at[i, prev_open_col]) else 0
					# if previous result says station already opened at this cluster-period,
					# this model must have x_t_i >= that previous value
					prob.addConstraint(x_t_i[i, t] == val)

				if prev_bikes_col in prev.columns:
					val = int(prev.at[i, prev_bikes_col]) if not pd.isna(prev.at[i, prev_bikes_col]) else 0
					prob.addConstraint(y_t_i[i, t] == val)

				if prev_hangars_col in prev.columns:
					val = int(prev.at[i, prev_hangars_col]) if not pd.isna(prev.at[i, prev_hangars_col]) else 0
					prob.addConstraint(z_t_i[i, t] == val)
		
	# total cost constraint
	prob.addConstraint(total_cost <= budget)
	# period cost constrain (which should also inherently enforce the total cost constraint but may as well keep both, no?)
	for t in periods:
		prob.addConstraint(period_costs[t] <= budget * periodic_pct_budget[t])
	# environmental constraint
	# prob.addConstraint(environmental_value >= ep*fuel_cost)
	enviro_rhs = 0
	for t in periods:
		enviro_rhs += ep*fuel_cost*periodic_pct_sec_obj[t]
	prob.addConstraint(environmental_value_total >= enviro_rhs)
	#prob.addConstraint(environmental_value_i >= ep * fuel_cost * x[i]) proportional to cost
	for i in clusters:
		prob.addConstraint(x[i] <= 1)
	# Budget limit exception
	# prob.addConstraint(total_cost <= l[1]*social_value + l[2]*environmental_value)

	# Minimum coverage
	# approximately 75% coverage of POIs
	coverage_t_rhs = 0
	for t in periods:
		coverage_t_rhs += 25000 * periodic_pct_sec_obj[t]
	prob.addConstraint(min_coverage >= coverage_t_rhs)
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

	prob.write("extension_bikes","lp")
	#%%
	xp.setOutputEnabled(True)
	prob.solve()
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
	results_frame.to_csv('extension_results.csv', index=False)

	return results_frame

#%%
def update_m(previous_results, m=None):
	# if m == None:
	# 	m = np.load('m_matrix.npy')
	# unpack previous results into arrays 
	x, y = unpack_previous_results(previous_results)
	I, T = x.shape

	# cumulative bikes for rule 1
	y_cumsum = y.cumsum(axis=1)   # shape (I, T)

	M_prev = m.copy()
	M_t = np.zeros_like(m)

	# loop over each time period (you update m sequentially)
	for t in range(T):
		open_rows = x[:, t]
		open_cols = open_rows   # same for rows/columns

		# Precompute masks
		both_open = np.outer(open_rows, open_cols)
		rowop_colclos = np.outer(open_rows, ~open_cols)
		rowclos_colop = np.outer(~open_rows, open_cols)
		both_closed = np.outer(~open_rows, ~open_cols)

		# Rule 1: both open
		num = y_cumsum[:, t-1][:,None] + y_cumsum[:, t-1][None,:]
		den = y_cumsum[:, t-2][:,None] + y_cumsum[:, t-2][None,:]
		safe_den = den.copy()
		safe_den[safe_den == 0] = 1   # avoid division by zero
		ratio = num / safe_den
		ratio[den == 0] = 1           # force no growth when no prior info
		mu = ratio * M_prev
		samples = np.random.normal(mu[both_open], 4)
		samples = np.maximum(samples, 0)        # clamp negatives to 0
		M_t[both_open] = np.rint(samples).astype(int)

		# Rule 2: i open, j closed
		avg_over_open_cols = M_prev[:, open_cols].mean(axis=1)  # shape = (I,)

		# Create full matrix version
		avg_matrix = np.tile(avg_over_open_cols[:, None], (1, I))  # shape = (I, I)

		M_t[rowop_colclos] = np.rint(avg_matrix[rowop_colclos]).astype(int)


		# Rule 3: i closed, j open
		avg_over_open_rows = M_prev[open_rows, :].mean(axis=0)  # shape = (I,)
		avg_matrix = np.tile(avg_over_open_rows[None, :], (I, 1))
		M_t[rowclos_colop] = np.rint(avg_matrix[rowclos_colop]).astype(int)

		# Rule 4: both closed
		avg_open_block = M_prev[np.ix_(open_rows, open_cols)].mean()
		M_t[both_closed] = np.rint(avg_open_block).astype(int)

		# update previous matrix
		M_prev = M_t.copy()

	
	# Save final updated matrix
	np.save('m_matrix.npy', M_prev)
	np.savetxt('m_matrix.csv', M_prev, delimiter=",")


	return M_prev
#%%
def unpack_previous_results(previous_results):
	periods = []
	for col in previous_results.columns:
		if col.startswith("opened during "):
			try:
				t = int(col.split()[-1])
				periods.append(t)
			except ValueError:
				continue

	periods = sorted(set(periods))  # e.g., [-1, 0, 1]


	for t in periods:
		if f"opened during {t}" not in previous_results.columns:
			raise ValueError(f"Missing column: opened during {t}")
		if f"bikes added during {t}" not in previous_results.columns:
			raise ValueError(f"Missing column: bikes added during {t}")


	I = previous_results.shape[0]
	T = len(periods)
	x = np.zeros((I, T), dtype=bool)
	y = np.zeros((I, T), dtype=float)


	for idx, t in enumerate(periods):
		x[:, idx] = previous_results[f"opened during {t}"].astype(bool).values
		y[:, idx] = previous_results[f"bikes added during {t}"].astype(float).values

	return x, y

#%%
previous_results = pd.DataFrame(np.zeros((250, 3), dtype = int), columns=["opened during 0", "bikes added during 0", "hangars added during 0"])
original_scheme = pd.DataFrame(np.zeros((250, 3), dtype = int), columns=["opened during -1", "bikes added during -1", "hangars added during -1"])
original_scheme.insert(0, "cluster", np.arange(250))
previous_results.insert(0, "cluster", np.arange(250))

station_assignments = pd.read_csv('stations_assigned.csv')
assigned_stations = station_assignments[station_assignments['assigned_cluster'] != -1]
station_to_cluster = dict(zip(assigned_stations['station_id'], assigned_stations['assigned_cluster']))

station_ids_in_matrix = sorted(assigned_stations['station_id'].tolist())
station_index_in_matrix = {station_id: idx for idx, station_id in enumerate(station_ids_in_matrix)}

curr_M = np.load('m_matrix.npy')

n = 250
cluster_m = np.zeros((n,n))

for i_station, i_cluster in station_to_cluster.items():
	for j_station, j_cluster in station_to_cluster.items():
		i_idx = i_cluster
		j_idx = j_cluster
		i_orig = station_index_in_matrix[i_station]
		j_orig = station_index_in_matrix[j_station]
		cluster_m[i_idx, j_idx] = curr_M[i_orig, j_orig]
		
# cluster_m[cluster_m==0] = m[cluster_m==0]

curr_M = cluster_m

station_assignments = pd.read_csv('stations_assigned.csv')
assigned_stations = station_assignments[station_assignments['assigned_cluster'] != -1]
station_to_cluster = dict(zip(assigned_stations['station_id'], assigned_stations['assigned_cluster']))

for _, row in assigned_stations.iterrows():
	cluster = int(row['assigned_cluster'])
	capacity = int(row['capacity'])

	original_scheme.loc[cluster, 'opened during -1'] = 1
	original_scheme.loc[cluster, 'bikes added during -1'] = capacity
			

for t in range(3):
	t_results = period_prob(previous_results, t, curr_M)
	new_and_old = original_scheme.merge(t_results, on = "cluster")
	curr_M = update_m(new_and_old, curr_M)
	previous_results = t_results
previous_results.to_csv('extension_results.csv', index=False)