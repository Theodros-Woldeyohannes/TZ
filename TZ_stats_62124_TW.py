# import packages

import os
import rasterio
from rasterio.mask import mask
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import geopandas as gpd

# Set base file path
file_path = r'D:\UNM\P50\Hudson_BGT_RWPR\Working_repo_61124\TZ'

# Read polygon files
polygon_BGT = gpd.read_file(file_path + '/Polygon/BGT_boundingBox.shp')
polygon_RWPR = gpd.read_file(file_path + '/Polygon/RWPR_boundingBox.shp')

# Plot polygon files
polygon_BGT.plot()
polygon_RWPR.plot()

# Get polygon geometry info
polygon_geometry_info_BGT = polygon_BGT.geometry.bounds
geometry_BGT = polygon_BGT.geometry.values[0]
print("Polygon geometry info:")
print(polygon_geometry_info_BGT)
print("Polygon bounds:", polygon_BGT.geometry.bounds)

polygon_geometry_info_RWPR = polygon_RWPR.geometry.bounds
geometry_RWPR = polygon_RWPR.geometry.values[0]
print("Polygon geometry info:")
print(polygon_geometry_info_RWPR)
print("Polygon bounds:", polygon_BGT.geometry.bounds)

# Define path to raster directory
raster_path = r'D:\UNM\P50\Hudson_BGT_RWPR\Working_repo_61124\TZ\Rasters/'

# Check and make sure path is set correctly
raster_files = os.listdir(raster_path)
for f in raster_files:
    print(str(f))

# Read the mock CSV file 
mock_csv_path = r"D:\UNM\P50\Hudson_BGT_RWPR\Working_repo_61124\TZ\TZ_mockData.csv"
mock_df = pd.read_csv(mock_csv_path)

# List of analytes
analytes = ['UTAS', 'UV', 'USE', 'UFE', 'UBE', 'UCO', 'USR', 'UMO', 'USN', 'USB', 'UCS', 'UBA', 'UW', 'UPT', 'UPB', 'UUR', 'UCD', 'UMN', 'SCU', 'SZN', 'STAS', 'SSE']

# Reduce the number of iterations for testing
num_iterations = 10  # Temporarily reduced for testing

# Perform statistical test for each analyte
for a in analytes:

    # Create a list to store the results
    sampled_data = []
    correlation_data = []

    # Loop through the process for each analyte
    for i in range(num_iterations):
        print(f"Starting iteration {i+1}")
        
        # Lists to collect values for correlation calculation
        values = []
        bio_results = []
        
        # Loop through each row in the DataFrame
        for index, row in mock_df.iterrows():
            print(f"Processing raster {row['raster']} (index {index})")
            # Open raster based on the name in the 'raster' column
            newraster = rasterio.open(os.path.join(raster_path, row['raster']))
            
            # Take random raster sample based on community:
            if row['community'] == 'BGT':

                # Create mask based on polygon file
                out_image, out_transform = mask(newraster, [geometry_BGT], crop=True)
                
                # Return exposure values from raster mask
                masked_data = out_image[0]
                
                # Clean the data
                masked_data = masked_data[~np.isnan(masked_data)]
                masked_data_clean = np.delete(masked_data, np.where(masked_data < 0))
                
                # Sample a random value from array and collect it, collect corresponding bio result
                sampled_value = np.random.choice(masked_data_clean, replace=False)
                values.append(sampled_value)
                bio_results.append(row[a])
            else:
                # Create mask based on polygon file
                out_image, out_transform = mask(newraster, [geometry_RWPR], crop=True)
                
                # Return exposure values from raster mask
                masked_data = out_image[0]
                
                # Clean the data
                masked_data = masked_data[~np.isnan(masked_data)]
                masked_data_clean = np.delete(masked_data, np.where(masked_data < 0))
                
                # Sample a random value from array and collect it, collect corresponding bio result
                sampled_value = np.random.choice(masked_data_clean, replace=False)
                values.append(sampled_value)
                bio_results.append(row[a])
        
        # Calculate Spearman correlation for the collected values and bio_results
        correlation = spearmanr(values, bio_results).correlation
        correlation_data.append({
            'Iteration': i + 1,
            'Correlation Value': correlation
        })
        
        # Append the sampled data for each participant for current iteration and analyte
        for index, row in mock_df.iterrows():
            sampled_data.append({
                'Iteration': i + 1,
                'Raster': row['raster'],
                a: row[a],
                'Sampled Value': values[index],
                'ID': row['ID'],
                'Community': row['community']
            })
        
        # Update counter
        print(f"Completed iteration {i+1}")

    # Choose a different directory to save the results
    results_dir = r'D:\UNM\P50\Hudson_BGT_RWPR\Working_repo_61124\TZ\results_test_4'
    os.makedirs(results_dir, exist_ok=True)

    # Create DataFrames from the collected sample data and correlaiton results
    sampled_df = pd.DataFrame(sampled_data)
    corr_df = pd.DataFrame(correlation_data)

    # Reorder the columns
    sampled_df = sampled_df[['Iteration', 'ID', 'Community', 'Raster', a, 'Sampled Value']]

    # Save results to CSV files
    csv_path_sd = os.path.join(results_dir, a + '_' + 'sampled_values.csv')
    sampled_df.to_csv(csv_path_sd, index=False)

    csv_path_cd = os.path.join(results_dir, a + '_' + 'correlation_values.csv')
    corr_df.to_csv(csv_path_cd, index=False)

    # Print a message indicating successful saving
    print(f"Sampled raster data saved to {csv_path_sd}")
    print(f"Correlation data saved to {csv_path_cd}")



