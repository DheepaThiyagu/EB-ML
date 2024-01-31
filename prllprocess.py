import pandas as pd
from appp import save_csv,delete_open_search_index,bulk_save_training_index,elasticsearch_host,elasticsearch_port,elk_username,elk_password

import pandas as pd

df = pd.read_csv('training/output_selected_dataframe.csv')

print( df.head())
df['fivemin_timestamp']=df['record_time']
# Assuming your DataFrame is named 'df'
df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')


# Extract hour and day from the 'record_time'
df['hour_timestamp'] = df['record_time'].dt.floor('H').astype('int64') // 10**6 # Hourly timestamp in milliseconds
df['day_timestamp'] = (df['record_time'].dt.floor('D')).astype('int64') // 10**6 # Daily timestamp in milliseconds

print(df)

df.to_csv('C:/Users/91701/EB_ML/training/new_training_data.csv', index=False)


# target_index='dheepa_csv_bulk'
target_index='new_hourly_daily_fivemins_753'
# # Load the CSV files into DataFrames
# df_1007 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1007_predicted.csv')
# df_1008 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1008_predicted.csv')
# df_1021 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1021_predicted.csv')
# df_1034 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1034_predicted.csv')
#
#
# # Concatenate DataFrames based on common columns
# combined_df = pd.concat([df_1007, df_1008,df_1021,df_1034], ignore_index=True, sort=False)
#
# # Save the combined DataFrame to a new CSV file
# combined_df.to_csv('C:/Users/91701/EB_ML/predictions/combined_data.csv', index=False)
#
# # Delete the old target OpenSearch index
# delete_open_search_index(target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)
#
bulk_save_training_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)


