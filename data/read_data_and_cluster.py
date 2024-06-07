import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import contextily as ctx
from sklearn.cluster import KMeans

# from https://zenodo.org/records/5146275
# data = pd.read_csv('./METR-LA.csv')
# data = data.rename(columns={'Unnamed: 0': 'timestamp'})
# data['timestamp'] = pd.to_datetime(data['timestamp'])
#
# # Set the 'timestamp' column as the index
# data = data.set_index('timestamp')
# # Display the first few rows of the data to understand its structure
# print("First few rows of the data:")
# print(data.head())
# print(len(data.columns))
#
# # Display statistical summary of the data
# print("\nStatistical summary of the data:")
# print(data.describe())
#
# # Read the CSV file
# df = pd.read_csv('./graph_sensor_locations.csv')
# # Create a GeoDataFrame and set the initial CRS to WGS84 (latitude/longitude in degrees)
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
#
# # Convert the GeoDataFrame to the Web Mercator projection (required by contextily)
# gdf = gdf.to_crs(epsg=3857)
#
# # Plot the GeoDataFrame
# fig, ax = plt.subplots(figsize=(16, 12))  # Adjust the figure size as needed
# gdf.plot(ax=ax, marker='o', color='black', markersize=40)
#
# # Add a basemap
# ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
# ax.set_title('METR-LA', size=20)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.savefig("./street_map.png")
# plt.show()

# -----------------------------------------CLUSTERING-----------------------------------------------------------


# Read the CSV file
df = pd.read_csv('./graph_sensor_locations.csv')
df = df.set_index('sensor_id')

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
gdf['cluster'] = kmeans.fit_predict(gdf[['geometry']].apply(lambda geom: geom.x))

fig, ax = plt.subplots(figsize=(16, 12))  # Adjust the figure size as needed

# Print indices for each cluster
for cluster_num, indices in gdf.groupby('cluster').groups.items():
    # Plot 5 values for the current cluster
    gdf_sample = gdf.loc[indices[:5]]  # Select the first 5 indices for the cluster
    gdf_sample.plot(ax=ax, color='red', markersize=100, label=f'Cluster {cluster_num} Sample Points', alpha=0.5)

    print(f"Cluster {cluster_num} Indices: {indices}")


# Plot the points with cluster coloring
gdf.plot(ax=ax, column='cluster', cmap='viridis', markersize=50, legend=True)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title('Clustered Locations Map', size=20)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
plt.savefig("./clustered_street_map.png")
plt.show()

