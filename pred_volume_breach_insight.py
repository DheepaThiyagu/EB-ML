import pandas as pd
import configparser
import os
from datetime import datetime, timedelta
import datetime as dt  # Import the datetime module with an alias
import pytz  # Import pytz library for timezone conversion


config = configparser.ConfigParser()
config.read('config.ini')

models_path = config.get('paths', 'models_path')
predictions_path = config.get('paths', 'predictions_path')
training_path = config.get('paths', 'training_path')
retraining_path = config.get('paths', 'retraining_path')

project_path = os.getcwd()

csv_file_path = os.path.join(project_path, predictions_path,'alldata_rf_clssfr_1model_brl_vol_per_service.csv')
df_breach = pd.read_csv(csv_file_path)
print(df_breach[['sid','breach_volume']])

def find_breach_times(sid):
    print(f"Predicted Volume breaches for {sid}  ")
    csv_file_path = os.path.join(project_path, predictions_path,f'{sid}_predicted.csv')
    df_predictions = pd.read_csv(csv_file_path)
    # print(df_predictions)

    # Filter rows based on the given sid
    df_breach_sid = df_breach[df_breach['sid'] == sid]
    df_predictions_sid = df_predictions[df_predictions['sid'] == sid]
    # print("after filter")
    # print(df_breach_sid)
    # print(df_predictions_sid)

    # Merge the two DataFrames on the timestamp column
    merged_df = pd.merge(df_predictions_sid, df_breach_sid, on='sid', how='inner')
    print(merged_df)

    #
    # # Find timestamps where pred_req_vol crosses the breach_volume
    # breach_times = merged_df[merged_df['pred_req_vol'] > merged_df['breach_volume']]['datetime']
    # print(breach_times)
    # # Print output in the desired format

    # Find timestamps where pred_req_vol crosses the breach_volume
    breach_times = merged_df[merged_df['pred_req_vol'] >= merged_df['breach_volume']]['Timestamp']


    for timestamp in breach_times:
            print(f"{timestamp} breach possible")

    # # Convert timestamps to datetime objects in UTC and then localize them to Asia/Manila timezone
    # manila_tz = pytz.timezone('Asia/Manila')
    # breach_timestamps = [dt.datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc).astimezone(manila_tz) for timestamp in breach_times]


    # Convert timestamps to datetime objects and format them(UTC time)
    breach_timestamps = [dt.datetime.fromtimestamp(timestamp / 1000) for timestamp in breach_times]

    # # Format the datetime objects as strings in the desired format
    # breach_timestamps = [timestamp.strftime('%Y-%m-%d %H:%M:%S') for timestamp in breach_timestamps]


# # Convert the breach timestamps to datetime objects
#     breach_timestamps = [datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in breach_times]

    # Initialize variables for storing the start and end timestamps of breach ranges
    start_time = breach_timestamps[0]
    end_time = breach_timestamps[0]

    # Initialize a list to store the text format of breach ranges
    breach_ranges = []


    # Initialize variables to track the current date
    current_date = breach_timestamps[0].date()

    # Iterate over the breach timestamps
    for timestamp in breach_timestamps:
        # Check if the date of the current timestamp is different from the current date
        if timestamp.date() != current_date:
            # Append the breach range for the previous date to the list
            # breach_ranges.append(f"{current_date.strftime('%Y-%m-%d')} {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} breach possible")
            breach_ranges.append(f"{current_date.strftime('%Y-%m-%d')} {start_time.strftime('%I:%M %p')} to {end_time.strftime('%I:%M %p')} breach possible")

        # Update the current date
            current_date = timestamp.date()
            # Update the start and end times for the next range
            start_time = timestamp
            end_time = timestamp
        elif timestamp == end_time + timedelta(minutes=5):
            # Update the end time of the current range
            end_time = timestamp
        else:
            # Append the breach range for the previous date to the list
            breach_ranges.append(f"{current_date.strftime('%Y-%m-%d')} {start_time.strftime('%I:%M %p')} to {end_time.strftime('%I:%M %p')} breach possible")
            # Update the start and end times for the next range
            start_time = timestamp
            end_time = timestamp

    # Append the last breach range to the list
            breach_ranges.append(f"{current_date.strftime('%Y-%m-%d')} {start_time.strftime('%I:%M %p')} to {end_time.strftime('%I:%M %p')} breach possible")

    # Join the breach ranges into a single text
    text_output = "\n".join(breach_ranges)

    # Print the combined text
    print(text_output)





sid = 16994  # Example sid
find_breach_times(sid)
