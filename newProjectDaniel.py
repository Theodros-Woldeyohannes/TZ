
# import os
# import rasterio
# from rasterio.mask import mask
# import pandas as pd
# import numpy as np
# from scipy.stats import spearmanr
# import geopandas as gpd

# # Set base file path
# file_path = r'C:\Users\mehdi\OneDrive\Bureau\Files_GitHUb_DanielProject\TZ'

# # Read polygon file
# polygon = gpd.read_file(file_path + '/Polygon/BGT_boundingBox.shp')

# # Plot polygon file
# polygon.plot()

# # Get polygon geometry info
# polygon_geometry_info = polygon.geometry.bounds
# geometry = polygon.geometry.values[0]
# print("Polygon geometry info:")
# print(polygon_geometry_info)
# print("Polygon bounds:", polygon.geometry.bounds)

# # Define path to raster directory
# raster_path = r'C:\Users\mehdi\OneDrive\Bureau\Files_GitHUb_DanielProject\TZ\Rasters/'

# # Check and make sure path is set correctly
# raster_files = os.listdir(raster_path)
# for f in raster_files:
#     print(str(f))

# # Read the mock CSV file to extract the ID information
# mock_csv_path = r"C:\Users\mehdi\OneDrive\Bureau\TZ_mockData.csv"
# mock_df = pd.read_csv(mock_csv_path)

# # Check if the lengths match and adjust accordingly
# num_raster_files = len(raster_files)
# if len(mock_df['ID']) >= num_raster_files:
#     ids = mock_df['ID'].iloc[:num_raster_files].tolist()
# else:
#     ids = mock_df['ID'].tolist() + list(range(len(mock_df['ID']) + 1, num_raster_files + 1))

# # Initialize the dataframe with IDs
# dataframe = pd.DataFrame({
#     'raster': raster_files,  # Assuming each raster file corresponds to a row
#     'bio_result': np.random.random(num_raster_files),  # Placeholder for the actual 'bio_result' values
#     'ID': ids  # Use the adjusted IDs
# })

# # Round the bio_result column to 2 decimal places
# dataframe['bio_result'] = dataframe['bio_result'].round(2)

# # Reduce the number of iterations for testing
# num_iterations = 10  # Temporarily reduced for testing

# # Create a list to store the results
# correlation_data = []

# # Loop through the process
# for i in range(num_iterations):
#     print(f"Starting iteration {i+1}")
    
#     # Lists to collect values for correlation calculation
#     values = []
#     bio_results = []
    
#     # Loop through each row in the DataFrame
#     for index, row in dataframe.iterrows():
#         print(f"Processing raster {row['raster']} (index {index})")
#         # Open raster based on the name in the 'raster' column
#         newraster = rasterio.open(os.path.join(raster_path, row['raster']))
        
#         # Create mask based on polygon file
#         out_image, out_transform = mask(newraster, [geometry], crop=True)
        
#         # Return exposure values from raster mask
#         masked_data = out_image[0]
        
#         # Clean the data
#         masked_data = masked_data[~np.isnan(masked_data)]
#         masked_data_clean = np.delete(masked_data, np.where(masked_data < 0))
        
#         # Sample a random value from array and collect it
#         sampled_value = np.random.choice(masked_data_clean, replace=False)
#         values.append(sampled_value)
#         bio_results.append(row['bio_result'])
    
#     # Calculate Spearman correlation for the collected values and bio_results
#     correlation = spearmanr(values, bio_results).correlation
    
#     # Append the results for each raster
#     for index, row in dataframe.iterrows():
#         correlation_data.append({
#             'Iteration': i + 1,
#             'Raster': row['raster'],
#             'Bio Result': row['bio_result'],
#             'Spearman Correlation': correlation,
#             'ID': row['ID']
#         })
    
#     # Update counter
#     print(f"Completed iteration {i+1}")

# # Choose a different directory to save the results
# results_dir = r'C:\Users\mehdi\OneDrive\Bureau\Files_GitHUb_DanielProject\TZ\results_alternative'
# os.makedirs(results_dir, exist_ok=True)

# # Create a DataFrame from the collected data
# final_df = pd.DataFrame(correlation_data)

# # Reorder the columns
# final_df = final_df[['Iteration', 'ID', 'Raster', 'Bio Result', 'Spearman Correlation']]

# # Save Spearman correlation results to a CSV file
# csv_path = os.path.join(results_dir, 'spearman_correlation_results_v2.csv')
# final_df.to_csv(csv_path, index=False)

# # Print a message indicating successful saving
# print(f"Spearman correlation results saved to {csv_path}")


from PIL import Image, ImageDraw, ImageOps

# Define the path to your image
img_path = r"C:\Users\mehdi\OneDrive\Bureau\Ucard pic.jpg"

# Load the image
img = Image.open(img_path)

# Create a copy of the original image to work on
edited_img = ImageOps.fit(img, img.size)

# Draw a white rectangle over the area where the tuxedo is
draw = ImageDraw.Draw(edited_img)
draw.rectangle([0, img.height // 2, img.width, img.height], fill="white")

# Draw a simple t-shirt shape
tshirt_color = (100, 100, 255)  # Blue color for the t-shirt
draw.rectangle([img.width // 4, img.height // 2, img.width * 3 // 4, img.height * 7 // 8], fill=tshirt_color)
draw.rectangle([img.width // 6, img.height // 2, img.width // 4, img.height * 5 // 8], fill=tshirt_color)
draw.rectangle([img.width * 3 // 4, img.height // 2, img.width * 5 // 6, img.height * 5 // 8], fill=tshirt_color)

# Define the path to save the edited image
edited_img_path = r"C:\Users\mehdi\OneDrive\Bureau\edited_Ucard_pic.jpg"
edited_img.save(edited_img_path)

print(f"Edited image saved at: {edited_img_path}")
