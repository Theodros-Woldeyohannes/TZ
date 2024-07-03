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

# OPTIONAL: Drop rows where 'visit' is 3 or 4
# data = data[~data['visit'].isin([3, 4])]

# List of analytes
analytes = ['UTAS', 'UV', 'USE', 'UFE', 'UBE', 'UCO', 'USR', 'UMO', 'USN', 'USB', 'UCS', 'UBA', 'UW', 'UPT', 'UPB', 'UUR', 'UCD', 'UMN', 'SCU', 'SZN', 'STAS', 'SSE']

# List to store spearman rank coefficients for each bootstrap iteration
bootstrap_results = []

# Dictionary to track the first print for each analyte
first_print = {analyte: True for analyte in analytes}

# Bootstrap n times
for _ in range(10000): # Change this
    # Create a new column 'rand_vals' in the data df
    data['rand_vals'] = np.nan

    # Write random values from extracted_values to rand_values based on dictionary key
    for index, row in data.iterrows():
        community = row['community']
        raster = row['raster']
        key = f"{community}_{raster}"
        
        if key in extracted_values:
            # Extract one random value 
            random_value = random.choice(extracted_values[key])
            data.at[index, 'rand_vals'] = random_value

    # Dictionary to store spearman rank coefficients for iteration
    spearman_results = {analyte: np.nan for analyte in analytes}

    # Spearman rank coefficient between 'rand_vals' and each analyte
    for analyte in analytes:
        analyte_values = data[analyte]
        
        # Drop NA 
        clean_data = data.dropna(subset=['rand_vals', analyte])
        
        # Count NA
        na_count = len(data) - len(clean_data)
        if first_print[analyte]:
            print(f'# NAs in {analyte}: {na_count}')
            first_print[analyte] = False
        
        if pd.api.types.is_numeric_dtype(analyte_values) and len(analyte_values.unique()) > 1:
            correlation, _ = spearmanr(clean_data['rand_vals'], clean_data[analyte])
            spearman_results[analyte] = correlation

    # Append each iteration to bootstrap_results 
    bootstrap_results.append(spearman_results)

# Convert bootstrap_results to df
bootstrap_df = pd.DataFrame(bootstrap_results)

# Save as csv
output_path = file_path + '/bootstrap_spearman_df.csv'
bootstrap_df.to_csv(output_path, index=False)
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
