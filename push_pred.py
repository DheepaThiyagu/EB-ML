import pandas as pd
from appp import save_csv,delete_open_search_index,bulk_save_forecast_index,elasticsearch_host,elasticsearch_port,elk_username,elk_password

df= pd.read_csv('C:/Users/91701/EB_ML/predictions/combined_forecasts.csv')

target_index='dheepa_753_prediction'

# Concatenate DataFrames based on common columns
# Extract hour and day from the 'record_time'
df['fivemin_timestamp']=df['Timestamp']

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

df['hour_timestamp'] = df['Timestamp'].dt.floor('H').astype('int64') // 10**6 # Hourly timestamp in milliseconds
df['day_timestamp'] = (df['Timestamp'].dt.floor('D')).astype('int64')// 10**6  # Daily timestamp in milliseconds
df['Timestamp']=df['Timestamp'].astype('int64')// 10**6
print(df)

# Save the combined DataFrame to a new CSV file
df.to_csv('C:/Users/91701/EB_ML/predictions/dheepa_753_pred.csv', index=False)

bulk_save_forecast_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)
