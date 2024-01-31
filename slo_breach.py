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
# Assuming df is your DataFrame containing the data

# Filter DataFrame for error breaches and response breaches
filtered_eb = df[df['is_eb_breached'] == 1]
filtered_resp = df[df['is_response_breached'] == 1]

# Group by app_id and sid for error breaches, and find the rows with the minimum and maximum total_req_count for each group
min_max_vol_eb = filtered_eb.groupby(['app_id', 'sid'])['total_req_count'].agg(['idxmin', 'idxmax']).reset_index()
min_max_vol_resp = filtered_resp.groupby(['app_id', 'sid'])['total_req_count'].agg(['idxmin', 'idxmax']).reset_index()

# Extract the rows with the minimum and maximum volume
min_vol_rows_eb = df.loc[min_max_vol_eb['idxmin']]
max_vol_rows_eb = df.loc[min_max_vol_eb['idxmax']]

min_vol_rows_resp = df.loc[min_max_vol_resp['idxmin']]
max_vol_rows_resp = df.loc[min_max_vol_resp['idxmax']]

# Concatenate the DataFrames for error breaches and response breaches
result_df_eb = pd.concat([min_vol_rows_eb, max_vol_rows_eb], axis=1)
result_df_resp = pd.concat([min_vol_rows_resp, max_vol_rows_resp], axis=1)

# Extract the required columns
result_df_eb = result_df_eb[['app_id', 'sid', 'record_time', 'total_req_count', 'is_eb_breached']]
result_df_resp = result_df_resp[['app_id', 'sid', 'record_time', 'total_req_count', 'is_response_breached']]

# Rename columns for clarity
result_df_eb.columns = ['app_id', 'sid', 'start_time', 'start_volume', 'is_eb_breached', 'app_id', 'sid', 'end_time', 'end_volume', 'is_eb_breached']
result_df_resp.columns = ['app_id', 'sid', 'start_time', 'start_volume', 'is_response_breached', 'app_id', 'sid', 'end_time', 'end_volume', 'is_response_breached']

# Save the results to CSV files
result_csv_path_eb = 'min_max_volume_breach_times_eb.csv'
result_csv_path_resp = 'min_max_volume_breach_times_resp.csv'
result_df_eb.to_csv(result_csv_path_eb, index=False)
result_df_resp.to_csv(result_csv_path_resp, index=False)

print(f"Error breach results saved to: {result_csv_path_eb}")
print(f"Response breach results saved to: {result_csv_path_resp}")
