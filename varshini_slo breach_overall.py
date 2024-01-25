
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


# Assuming thresholds for SLO breaches (these can be adjusted as per actual requirements)
response_slo_threshold = 1  # Assuming any non-zero value in 'is_response_breached' indicates a breach


# Convert 'record_time' to a readable date and time format
data['record_time'] = pd.to_datetime(data['record_time'], unit='ms')
data['date'] = data['record_time'].dt.date
data['day_of_week'] = data['record_time'].dt.day_name()
data['time'] = data['record_time'].dt.time


# Analyze for Response SLO Breaches
response_breaches = data[(data['is_response_breached'] == response_slo_threshold)]
response_breaches_summary = response_breaches.groupby(['date', 'day_of_week']).agg(
    time_range=('time', lambda x: (min(x), max(x))),
    total_req=('total_req_count', 'sum')
)

# Analyze for Error SLO Breaches
error_breaches = data[data['is_eb_breached'] == 1]
error_breaches_summary = error_breaches.groupby(['date', 'day_of_week']).agg(
    time_range=('time', lambda x: (min(x), max(x))),
    total_req=('total_req_count', 'sum')
)
print(error_breaches_summary)

# Generating descriptive lines for both types of breaches
response_slo_lines = [f"Response SLO breaches beyond {row['total_req']} volume requests, the observed day and time is on {idx[1]} between {row['time_range'][0].strftime('%H:%M:%S')} and {row['time_range'][1].strftime('%H:%M:%S')}." for idx, row in response_breaches_summary.iterrows()]
error_slo_lines = [f"Error SLO breaches beyond {row['total_req']} volume requests on {idx[1]} between {row['time_range'][0].strftime('%H:%M:%S')} and {row['time_range'][1].strftime('%H:%M:%S')}." for idx, row in error_breaches_summary.iterrows()]

# Adjusting the time format to include AM/PM
response_breaches_summary['time_range'] = response_breaches_summary['time_range'].apply(
    lambda x: (x[0].strftime('%I:%M %p'), x[1].strftime('%I:%M %p'))
)
error_breaches_summary['time_range'] = error_breaches_summary['time_range'].apply(
    lambda x: (x[0].strftime('%I:%M %p'), x[1].strftime('%I:%M %p'))
)

# Generating descriptive lines with AM/PM format
response_slo_lines_am_pm = [
    f"Response SLO breaches beyond {row['total_req']} volume requests, "
    f"the observed day and time is on {idx[1]} between {row['time_range'][0]} and {row['time_range'][1]}."
    for idx, row in response_breaches_summary.iterrows()
]

error_slo_lines_am_pm = [
    f"Error SLO breaches beyond {row['total_req']} volume requests on {idx[1]} "
    f"between {row['time_range'][0]} and {row['time_range'][1]}."
    for idx, row in error_breaches_summary.iterrows()
]

# Printing the result lines for Response and Error SLO breaches
for line in response_slo_lines_am_pm:
    print(line)

for line in error_slo_lines_am_pm:
    print(line)



# Printing the result lines for Response and Error SLO breaches
#for line in response_slo_lines:
#   print(line)

#for line in error_slo_lines:
#   print(line)

# Plotting graphs for Response and Error SLO breaches
plt.figure(figsize=(15, 6))

# Plot for Response SLO breaches
plt.subplot(1, 2, 1)
sns.barplot(x='date', y='total_req', data=response_breaches_summary.reset_index(), palette='Blues')
plt.title('Response SLO Breaches by Date')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Total Volume of Requests')

# Plot for Error SLO breaches
plt.subplot(1, 2, 2)
sns.barplot(x='date', y='total_req', data=error_breaches_summary.reset_index(), palette='Reds')
plt.title('Error SLO Breaches by Date')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Total Volume of Requests')

plt.tight_layout()
plt.show()
