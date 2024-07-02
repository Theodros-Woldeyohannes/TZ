# import packages

import os
import rasterio
from rasterio.mask import mask
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import geopandas as gpd
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Read in data
# Set base file path
file_path = r'G:\Hudson_BGT_RWPR\RandomSampling\TZ'

# Import CSV
data_path = file_path + '/TZ_mockData.csv'
data = pd.read_csv(data_path)

# Read polygon files
polygon_BGT = gpd.read_file(file_path + '/Polygon/BGT_boundingBox.shp')
polygon_RWPR = gpd.read_file(file_path + '/Polygon/RWPR_boundingBox.shp')

# Polygon geometry
geometry_BGT = polygon_BGT.geometry.values[0]
geometry_RWPR = polygon_RWPR.geometry.values[0]

# Read raster files
raster_path = file_path + '/Rasters'
raster_files = os.listdir(raster_path)

print(data.head())

# Dictionary to store the extracted values
extracted_values = {}

# Loop through rasters to extract by mask and store raster values in lists
for raster_file in raster_files:
    raster_name = os.path.splitext(raster_file)[0]
    raster_full_path = os.path.join(raster_path, raster_file)
    
    with rasterio.open(raster_full_path) as src:
        # Extract values for BGT community
        out_image_BGT, out_transform_BGT = mask(src, [geometry_BGT], crop=True)
        values_BGT = np.array(out_image_BGT).flatten()
        values_BGT = values_BGT[values_BGT != src.nodata]  # Remove no data values
        list_name_BGT = f"BGT_{raster_name}"
        extracted_values[list_name_BGT] = values_BGT.tolist()
        
        # Extract values for RWPR community
        out_image_RWPR, out_transform_RWPR = mask(src, [geometry_RWPR], crop=True)
        values_RWPR = np.array(out_image_RWPR).flatten()
        values_RWPR = values_RWPR[values_RWPR != src.nodata]  # Remove no data values
        list_name_RWPR = f"RWPR_{raster_name}"
        extracted_values[list_name_RWPR] = values_RWPR.tolist()

# Define the list of analytes
analytes = ['UTAS', 'UV', 'USE', 'UFE', 'UBE', 'UCO', 'USR', 'UMO', 'USN', 'USB', 'UCS', 'UBA', 'UW', 'UPT', 'UPB', 'UUR', 'UCD', 'UMN', 'SCU', 'SZN', 'STAS', 'SSE']

# Initialize a list to store the spearman rank coefficients for each bootstrap iteration
bootstrap_results = []

# Run the bootstrapping 1000 times
for _ in range(1000):
    # Create a new column 'rand_vals' in the data DataFrame
    data['rand_vals'] = np.nan

    # Populate 'rand_vals' with random values from the corresponding dictionary in extracted_values
    for index, row in data.iterrows():
        community = row['community']
        raster = row['raster']
        key = f"{community}_{raster}"
        
        if key in extracted_values:
            # Extract one random value from the list
            random_value = random.choice(extracted_values[key])
            data.at[index, 'rand_vals'] = random_value

    # Initialize a dictionary to store the spearman rank coefficients for this iteration
    spearman_results = {analyte: np.nan for analyte in analytes}

    # Calculate the spearman rank coefficient between 'rand_vals' and each analyte
    for analyte in analytes:
        analyte_values = data[analyte]
        if pd.api.types.is_numeric_dtype(analyte_values) and len(analyte_values.unique()) > 1:
            correlation, _ = spearmanr(data['rand_vals'], analyte_values)
            spearman_results[analyte] = correlation

    # Append the results of this iteration to the bootstrap_results list
    bootstrap_results.append(spearman_results)

# Convert the bootstrap_results to a DataFrame for easier analysis
bootstrap_df = pd.DataFrame(bootstrap_results)

# Save the DataFrame to a CSV file
output_path = file_path + '/bootstrap_spearman_df.csv'
bootstrap_df.to_csv(output_path, index=False)

# Confirm that the file has been saved
output_path

print(bootstrap_df.head())

# Plot histograms for each column in bootstrap_df
for analyte in bootstrap_df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_df[analyte], kde=True)
    plt.title(f'Histogram of Spearman Coefficients for {analyte}')
    plt.xlabel('Spearman Coefficient')
    plt.ylabel('Frequency')
    
    # Save the plot as a jpg file
    output_plot_path = file_path + f'/Histogram_{analyte}.jpg'
    plt.savefig(output_plot_path, format='jpg')
    
    #plt.close()  
