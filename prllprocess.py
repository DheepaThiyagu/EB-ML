import pandas as pd
from appp import delete_open_search_index,bulk_save_forecast_index,elasticsearch_host,elasticsearch_port,elk_username,elk_password
target_index='dheepa_csv_bulk'
# Load the CSV files into DataFrames
df_1007 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1007_predicted.csv')
df_1008 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1008_predicted.csv')
df_1021 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1021_predicted.csv')
df_1034 = pd.read_csv('C:/Users/91701/EB_ML/predictions/1034_predicted.csv')


# Concatenate DataFrames based on common columns
combined_df = pd.concat([df_1007, df_1008,df_1021,df_1034], ignore_index=True, sort=False)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('C:/Users/91701/EB_ML/predictions/combined_data.csv', index=False)

# Delete the old target OpenSearch index
delete_open_search_index(target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

# bulk_save_forecast_index(combined_df, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)
