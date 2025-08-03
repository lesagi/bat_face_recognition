import os

import pandas as pd

print(os.listdir('../face_recognition-no_background_05_07_24/data'))
# Load the data from the CSV file
filename = 'predictions_20250406_055250'
file_path = f'/Users/MAC/Documents/bat_face_rec/face_recognition-with_bg_31_03_25/{filename}.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Group by input1_name and input1_frame and find the input2_name with the highest y_hat rate
result = df.groupby(['input1_name', 'input1_frame']).apply(
    lambda group: group.loc[group['y_hat'].idxmax()]
).reset_index(drop=True)

# Select only the relevant columns
result = result[['input1_name', 'input1_frame', 'input2_name', 'y_hat']]

# Display the result
print(result)

# Save the result to a new CSV file if needed
result.to_csv(f'{filename}-highest_y_hat_per_combination.csv', index=False)
