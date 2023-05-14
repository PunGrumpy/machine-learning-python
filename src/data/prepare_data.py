import pandas as pd
import numpy as np
import os

# Define the directory and file
dir_path = "data_raw"
file_path = os.path.join(dir_path, "raw_data.csv")

# Create the directory if it doesn't exist
os.makedirs(dir_path, exist_ok=True)

# Define the dataframe
df = pd.DataFrame(
    data=np.random.rand(100, 5),  # 100 rows, 5 columns
    columns=["Feature1", "Feature2", "Feature3", "Feature4", "Target"],  # Column names
)

# Save the dataframe to a CSV file
df.to_csv(file_path, index=False)
