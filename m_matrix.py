import pandas as pd
import numpy as np

trip_data = pd.read_csv('all_years_start_to_end.csv')
# filltering by month and year
used_month = trip_data[(trip_data['year'] == 2020) & (trip_data['month'] == 1)]
# grouping trips based on start and end station id
grouped_trips = (
    used_month.groupby(['start_station_id', 'end_station_id'])['num_trips']
    .sum()
    .reset_index()
)

grouped_trips['start_station_id'] = grouped_trips['start_station_id'].astype(str).str.strip().astype(int)
grouped_trips['end_station_id'] = grouped_trips['end_station_id'].astype(str).str.strip().astype(int)

# mapping station ids to 1-31
unique_stations = sorted(set(grouped_trips['start_station_id']) | set(grouped_trips['end_station_id']))
station_map = {old_id: new_id for new_id, old_id in enumerate(unique_stations, start=1)}
grouped_trips['start_mapped'] = grouped_trips['start_station_id'].map(station_map)
grouped_trips['end_mapped'] = grouped_trips['end_station_id'].map(station_map)

# Creating the matrix
matrix_size = len(unique_stations)
trip_matrix = np.zeros((matrix_size, matrix_size))

# Filling the matrix
for row in grouped_trips.itertuples(index=False):
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

n = 250
m = np.zeros((n, n), dtype=int)
average_rows=[0]*n
average_columns=[0]*n
for i in range(0,250):
    for j in range(0,250):
        if (i<=96 and j<=96):
            m[i,j]=trip_matrix[i,j]
            average_columns[i]=average_columns[i]+m[i,j]
            average_rows[j]=average_rows[j]+m[i,j]
        if (i<=96 and j>96):
            m[i,j]=average_columns[i]/97
        if (i>96 and j<=96):
            m[i,j]=average_rows[j]/97
        if (i>96 and j>96):
            m[i,j]=sum(average_columns)/(97**2)
            
np.save('m_matrix.npy', m)

print(m)
pd.DataFrame(m).to_csv("full_matrix.csv", index=False)
print("Matrix saved as full_matrix.csv ✅")
