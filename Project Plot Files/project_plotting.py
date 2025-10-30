import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = r"C:\Users\toddh\Documents\aaaa Masters\MMCS\cyclehire-data\cyclehire-cleandata\all_years_trips_by_month.csv"
OUTPUT_DIR = r"C:\Users\toddh\Documents\aaaa Masters\MMCS\cyclehire-data\cyclehire-cleandata"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
print("Loaded", len(df), "rows")

# a couple hundred data points are listed as 1970, I just
#removed them here, but perhaps theres a better way to fix them
df = df[df['year'] >= 2000].copy()
df['month'] = df['month'].astype(int)

#save the new data without 1970 to a file to pull into qgis
cleaned_file = os.path.join(OUTPUT_DIR, "all_years_trips_clean.csv")
df.to_csv(cleaned_file, index=False)


#plot yearly totals of rides
yearly = df.groupby('year')['num_trips'].sum().reset_index()
yearly['pct_change'] = yearly['num_trips'].pct_change() * 100
yearly.to_csv(os.path.join(OUTPUT_DIR, "yearly_totals.csv"), index=False)

plt.figure(figsize=(8,5))
plt.plot(yearly['year'], yearly['num_trips'], marker='o')
plt.title("Total Trips per Year")
plt.xlabel("Year")
plt.ylabel("Total Trips")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_yearly_totals.png"))
plt.close()

#plot monthly averages of rides
monthly_avg = df.groupby('month')['num_trips'].mean().reset_index()
monthly_avg.to_csv(os.path.join(OUTPUT_DIR, "monthly_average_trips.csv"), index=False)

plt.figure(figsize=(8,5))
plt.plot(monthly_avg['month'], monthly_avg['num_trips'], marker='o')
plt.title("Average Monthly Trips (Across Years)")
plt.xlabel("Month")
plt.ylabel("Average Trips")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_monthly_average.png"))
plt.close()

#plot the top 10 most popular bike stations
station_totals = df.groupby(['start_station_id','start_station_name'])['num_trips'].sum().reset_index()
station_totals = station_totals.sort_values('num_trips', ascending=False)
station_totals.to_csv(os.path.join(OUTPUT_DIR, "station_totals.csv"), index=False)

top10 = station_totals.head(10)
plt.figure(figsize=(9,5))
plt.barh(top10['start_station_name'], top10['num_trips'])
plt.title("Top 10 Busiest Stations (All Years)")
plt.xlabel("Total Trips")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_top10_stations.png"))
plt.close()

#plot trips of the top 5 stations by year
pivot = df.pivot_table(values='num_trips',
                       index='start_station_name',
                       columns='year',
                       aggfunc='sum',
                       fill_value=0)
pivot.to_csv(os.path.join(OUTPUT_DIR, "station_yearly_pivot.csv"))

top5 = top10['start_station_name'][:5]
plt.figure(figsize=(9,5))
for s in top5:
    plt.plot(pivot.columns, pivot.loc[s], marker='o', label=s)
plt.title("Top 5 Station Trends by Year")
plt.xlabel("Year")
plt.ylabel("Trips")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_top5_station_trends.png"))
plt.close()

#plot trips by month for each year
monthly_by_year = df.groupby(['year', 'month'])['num_trips'].sum().reset_index()
plt.figure(figsize=(9,5))
for y in sorted(monthly_by_year['year'].unique()):
    subset = monthly_by_year[monthly_by_year['year'] == y]
    plt.plot(subset['month'], subset['num_trips'], marker='o', label=str(y))
plt.title("Monthly Trip Totals by Year")
plt.xlabel("Month")
plt.ylabel("Trips")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_monthly_by_year.png"))
plt.close()

