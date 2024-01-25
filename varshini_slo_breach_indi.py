import pandas as pd
import matplotlib.pyplot as plt


# Function to input service ID
def input_service_id():
    try:
        service_id = int(input("Enter the service ID: "))
        return service_id
    except ValueError:
        print("Please enter a valid integer for the service ID.")
        return None

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

csv_file_path = os.path.join(project_path, training_path, 'output_selected_dataframe.csv')
# csv_file_path = os.path.join(project_path, predictions_path, 'combined_forecasts.csv')
data = pd.read_csv(csv_file_path)

# Convert 'record_time' to a readable datetime format
data['datetime'] = pd.to_datetime(data['record_time'], unit='ms')

# Input service ID
service_id = input_service_id()
if service_id is not None:
    # Filter data for the input service ID
    data_filtered = data[data['sid'] == service_id]

    # Grouping by 2-hour intervals
    grouped_data = data_filtered.groupby([pd.Grouper(key='datetime', freq='1H')]).agg(
        total_error_breaches=('is_eb_breached', 'sum'),
        total_response_breaches=('is_response_breached', 'sum'),
        total_requests=('total_req_count', 'sum')
    ).reset_index()

    # Plotting the time series line graphs
    plt.figure(figsize=(15, 6))

    plt.plot(grouped_data['datetime'], grouped_data['total_response_breaches'], label='Response SLO Breaches',
             color='blue')
    plt.plot(grouped_data['datetime'], grouped_data['total_error_breaches'], label='Error SLO Breaches', color='red')

    plt.xlabel('Date and Time')
    plt.ylabel('Total Breaches')
    plt.title(f'Response and Error SLO Breaches Over Time for Service ID {service_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    # Initializing flags to check if any breaches occurred
    response_breach_occurred = False
    error_breach_occurred = False

    # Generating summary lines
    for index, row in grouped_data.iterrows():
        if row['total_response_breaches'] > 0:
            response_breach_occurred = True
            print(
                f"For Service ID {service_id}, Response SLO is observed between {row['datetime'].strftime('%Y-%m-%d %I:%M %p')} and {(row['datetime'] + pd.Timedelta(hours=2)).strftime('%I:%M %p')} with {row['total_requests']} volume requests.")
        if row['total_error_breaches'] > 0:
            error_breach_occurred = True
            print(
                f"For Service ID {service_id}, Error SLO is observed between {row['datetime'].strftime('%Y-%m-%d %I:%M %p')} and {(row['datetime'] + pd.Timedelta(hours=2)).strftime('%I:%M %p')} with {row['total_requests']} volume requests.")

    # Checking if no breaches occurred
    if not response_breach_occurred:
        print(f"No Response SLO breaches observed for Service ID {service_id}.")
    if not error_breach_occurred:
        print(f"No Error SLO breaches observed for Service ID {service_id}.")
else:
    print("No valid service ID provided.")