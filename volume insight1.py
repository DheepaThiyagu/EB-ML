import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
project_path = os.getcwd()

# Specify the relative path to your models directory
models_path = 'models'
predictions_path='predictions'
training_path='training'
retraining_path='retraining'



def print_volume_surge_time_ranges(csv_file_path,vol,dt,data,service_id):
    df = pd.read_csv(csv_file_path)

    if not service_id==None:
        # Filter data for the specified service ID
        service_data = df[df['sid'] == service_id]
    else:
        service_data=df

    if service_data.empty:
        print(f"No data found for Service ID {service_id}.")
        return

    # Calculate the threshold as 75% of the maximum volume
    volume_threshold = service_data[vol].max() * 0.75
    if not service_id==None:
        print(f"{data} Volume surge thersold for {service_id} :",volume_threshold)
    else:
        print(f"{data} Volume surge thersold for overall services :",volume_threshold)
    # Filter rows where 'total_req_count' is above the calculated threshold
    surges = service_data[service_data[vol] > volume_threshold]

    if surges.empty:
        print(f"No volume surges found for Service ID {service_id}.")
        return

    summary_of_surges(surges,dt,data,service_id)
    # Sort surges by volume in descending order
    # surges = surges.sort_values(by='total_req_count', ascending=False)
    # Calculate day of the week before sorting
    surges['day_of_week'] = pd.to_datetime(surges[dt], unit='ms').dt.strftime('%A')

    surges = surges.sort_values(by=[dt], ascending=[True])

    # # Print time ranges during which there is a volume surge( 5 mins)
    # print(f"Volume surges for Service ID {service_id} occur during the following time ranges:")
    # for _, row in surges.iterrows():
    #     start_time = pd.to_datetime(row['dt'], unit='ms')
    #     end_time = start_time + pd.Timedelta(minutes=5)  # Assuming 5-minute intervals, adjust as needed
    #     # day_of_week = start_time.strftime('%A')  # Extract day of the week
    #     day_of_week = row['day_of_week']
    #
    #     volume = row[vol]
    #     print(f"{day_of_week}-{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}, Volume: {volume}")

    # Print time ranges during which there is a volume surge(cumulative)
    print(f"{data} Volume surges for Service ID {service_id} occur during the following time ranges:")
    current_range_start = None
    current_range_end = None
    current_volume = None

    for _, row in surges.iterrows():
        #
        current_time = pd.to_datetime(row[dt], unit='ms')

        if current_range_end is not None and current_time - current_range_end <= pd.Timedelta(minutes=5):
            # Continue the current range
            current_range_end = current_time
        else:
            # Print the previous range (if any) and start a new one
            if current_range_start is not None:
                print_range(current_range_start, current_range_end, current_volume)
            current_range_start = current_time
            current_range_end = current_time
            current_volume = row[vol]

    # Print the last time range if there's one
    if current_range_start is not None:
        print_range(current_range_start, current_range_end, current_volume)


    # Calculate day of the week and hour for each surge (print surge summary)
    # prints only day -time  and surge count ex:Friday-07:13:20, Surge Count: 1
def summary_of_surges(surges,dt,data,service_id):
    surges['day_of_week'] = pd.to_datetime(surges[dt], unit='ms').dt.strftime('%A')
    surges['hour_of_day'] = pd.to_datetime(surges[dt], unit='ms').dt.strftime('%I:%M:%S %p')

    # Group surges by day and print the summary
    surge_summary = surges.groupby(['day_of_week', 'hour_of_day']).size().reset_index(name='surge_count')
    surge_summary = surge_summary.sort_values(by=['day_of_week', 'hour_of_day'])

    print(surge_summary)
    # Print the summary
    if service_id==None:
        print(f"{data} Volume surges for overall services occur during the following time ranges:")
    else:
        print(f"{data} Volume surges for Service ID {service_id} occur during the following time ranges:")

    surge_times = []  # List to store surge times for generalized printing

    for _, row in surge_summary.iterrows():
        print(f"{row['day_of_week']}-{row['hour_of_day']}, Surge Count: {row['surge_count']}")

        # Add the surge time to the list for generalized printing
        pd.to_datetime(surges[dt], unit='ms').dt.strftime('%I:%M:%S %p')
        surge_time = pd.to_datetime(row['hour_of_day'], format='%I:%M:%S %p')

        surge_times.append(surge_time)

    # Print the generalized time range
    if surge_times:
        print_range_summary(surge_times,service_id)

def print_range_summary(surge_times,service_id):
    start_time = min(surge_times)
    end_time = max(surge_times)
    if service_id== None:
        print(f"Volume surge for overall services happens between {start_time.strftime('%I:%M:%S %p')} to {end_time.strftime('%I:%M:%S %p')}")

    else:
        print(f"Volume surge for {service_id} happens between {start_time.strftime('%I:%M:%S %p')} to {end_time.strftime('%I:%M:%S %p')}")

def print_range(start_time, end_time, volume):
    print(f"{start_time.strftime('%A')}-{start_time.strftime('%Y-%m-%d %I:%M:%S %p')} to {end_time.strftime('%A')}-{end_time.strftime('%Y-%m-%d %I:%M:%S %p')}, Volume: {volume}")


def analyze_volume_patterns(csv_file_path,vol,dt,data,service_id):
    # Convert 'record_time' to datetime
    df = pd.read_csv(csv_file_path)

    if not service_id==None:
        # Filter data for the specified service ID
        service_data = df[df['sid'] == service_id]
    else:
        service_data=df

    service_data[dt] = pd.to_datetime(service_data[dt], unit='ms')

    # Extract day of the week and hour of the day
    service_data['day_of_week'] = service_data[dt].dt.day_name()
    service_data['hour_of_day'] = service_data[dt].dt.hour

    # Group by day of the week and hour of the day
    grouped_data = service_data.groupby(['day_of_week', 'hour_of_day'])[vol].mean().reset_index()

    # Pivot the data for better visualization
    pivot_data = grouped_data.pivot(index='day_of_week', columns='hour_of_day', values=vol)

    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt=".1f", cbar_kws={'label': 'Average Volume'})
    if not service_id==None:
        plt.title(f'Average Volume Patterns of {data} for {service_id}')
    else:
        plt.title(f'Average Volume Patterns of {data} for overall services')

    plt.show()

print("Enter an option(1 or 2) 1. for individual service\n 2. for overall services")
option=int(input("enter option"))
if option==1:
    selected_service_id = int(input("Enter the service ID: "))  # User enters the service ID
else:
    selected_service_id=None

training_file_path = os.path.join(project_path, training_path, 'output_selected_dataframe.csv')
prediction_file_path = os.path.join(project_path, predictions_path, 'combined_forecasts.csv')

print_volume_surge_time_ranges(training_file_path,'total_req_count','record_time','Training data',selected_service_id)
print_volume_surge_time_ranges( prediction_file_path, 'pred_req_vol', 'Timestamp','Prediction data',selected_service_id)

# print_volume_surge_time_ranges(training_file_path,'total_req_count','record_time','Training data',selected_service_id)
# print_volume_surge_time_ranges( prediction_file_path, 'pred_req_vol', 'Timestamp','Prediction data',selected_service_id)


# Example: Analyze volume patterns in the entire dataset
analyze_volume_patterns(training_file_path,'total_req_count','record_time','Training data',selected_service_id)
analyze_volume_patterns(prediction_file_path,'pred_req_vol','Timestamp','Prediction data',selected_service_id)

