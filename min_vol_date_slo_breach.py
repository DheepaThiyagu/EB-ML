import pandas as pd
import pandas as pd
import os
project_path = os.getcwd()

# Specify the relative path to your models directory
models_path = 'models'
predictions_path='predictions'
training_path= 'training'
retraining_path='retraining'

csv_file_path = os.path.join(project_path, training_path, 'output_selected_dataframe.csv')
df = pd.read_csv(csv_file_path)

# Assuming df is your DataFrame
# Filter based on is_eb_breached and is_response_breached
filtered_eb = df[df['is_eb_breached'] == 1]
filtered_resp = df[df['is_response_breached'] == 1]

# Group by app_id and sid, and calculate the minimum total_req_count and time ranges for each group
min_vol_and_time_for_eb = filtered_eb.groupby(['app_id', 'sid']).agg({
    'total_req_count': 'min',
    'record_time': ['min', 'max']
}).reset_index()

min_vol_and_time_for_resp = filtered_resp.groupby(['app_id', 'sid']).agg({
    'total_req_count': 'min',
    'record_time': ['min', 'max']
}).reset_index()

# Rename columns for clarity
min_vol_and_time_for_eb.columns = ['app_id', 'sid', 'min_vol_for_eb_breach', 'start_time_for_eb_breach', 'end_time_for_eb_breach']
min_vol_and_time_for_resp.columns = ['app_id', 'sid', 'min_vol_for_response_breach', 'start_time_for_response_breach', 'end_time_for_response_breach']


# Convert timestamp columns to datetime format
min_vol_and_time_for_eb['start_time_for_eb_breach'] = pd.to_datetime(min_vol_and_time_for_eb['start_time_for_eb_breach'], unit='ms')
min_vol_and_time_for_eb['end_time_for_eb_breach'] = pd.to_datetime(min_vol_and_time_for_eb['end_time_for_eb_breach'], unit='ms')

min_vol_and_time_for_resp['start_time_for_response_breach'] = pd.to_datetime(min_vol_and_time_for_resp['start_time_for_response_breach'], unit='ms')
min_vol_and_time_for_resp['end_time_for_response_breach'] = pd.to_datetime(min_vol_and_time_for_resp['end_time_for_response_breach'], unit='ms')

# Merge the two DataFrames on app_id and sid
result_df = pd.merge(min_vol_and_time_for_eb, min_vol_and_time_for_resp, on=['app_id', 'sid'], how='outer')

# Sort by 'sid'
result_df = result_df.sort_values(by='sid')

# Save the result to a CSV file
result_csv_path = os.path.join(project_path, 'min_vol_and_time_for_breache.csv')
result_df.to_csv(result_csv_path, index=False)

print(f"Result saved to: {result_csv_path}")
