import pandas as pd

poi_data = pd.read_csv('enbraPOIClusterAssingments.csv')

poi_data.columns = poi_data.columns.str.strip()

poi_data = poi_data[['assigned cluster', 'category']]
poi_data['assigned cluster'] = poi_data['assigned cluster'].astype(int)
poi_data['category'] = poi_data['category'].astype(str).str.strip()

building_numbers = (
    poi_data.groupby(['assigned cluster', 'category'])
    .size()
    .reset_index(name='count'))

a_dict = {(row['assigned cluster'], row['category']): row['count'] 
     for _, row in building_numbers.iterrows()}

def a(i,k):
    return a_dict.get((i,k), 0)



