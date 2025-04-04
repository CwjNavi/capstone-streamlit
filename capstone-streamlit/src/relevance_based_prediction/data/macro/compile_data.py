import os

import pandas as pd

# Define the folder path
folder_path = "Relevance Based Prediction/data/macro/FRED"

# Initialize an empty DataFrame to store the merged result
result = None

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # adjust file extension if needed
        # Read each file into a DataFrame, interpreting "." as NaN
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, na_values=".")
        
        # Ensure that the "DATE" column is treated as datetime if not already
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        
        # Merge on "DATE" column
        if result is None:
            result = df  # initialize with the first DataFrame
        else:
            result = pd.merge(result, df, on='DATE', how='outer')

histimpl_df = pd.read_excel("Relevance Based Prediction/data/macro/histimpl.xls", skiprows=6, skipfooter=8)
# Convert the 'Year' column to a date representing January 1st of that year
histimpl_df['DATE'] = pd.to_datetime(histimpl_df['Year'].astype(str) + '-01-01')

result = pd.merge(result, histimpl_df[["DATE", "Implied ERP (FCFE)"]], on='DATE', how='outer')

# Display the result
print(result)
result.to_csv("Relevance Based Prediction/data/macro/macro.csv", index=False)