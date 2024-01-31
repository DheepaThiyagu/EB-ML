import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
project_path = os.getcwd()

# Specify the relative path to your models directory
models_path = 'models'
predictions_path='predictions'
training_path= 'training'
retraining_path='retraining'

csv_file_path = os.path.join(project_path, training_path, 'output_selected_dataframe.csv')
data = pd.read_csv(csv_file_path)

# Convert 'record_time' from UNIX timestamp to datetime
data['datetime'] = pd.to_datetime(data['record_time'], unit='ms')

# Extract day of the week and hour from the datetime
data['day_of_week'] = data['datetime'].dt.day_name()
data['hour'] = data['datetime'].dt.hour

# User input for service ID
selected_service_id = int(input("Enter the service ID: "))  # User enters the service ID

# Filter data for the selected service ID
service_data = data[data['sid'] == selected_service_id]

# Group by day of the week and hour for the selected service
grouped_data = service_data.groupby(['day_of_week', 'hour']).agg({'total_req_count': 'mean'}).reset_index()

# Finding the time ranges with the highest average request count for each day
max_req_per_day = grouped_data.loc[grouped_data.groupby(['day_of_week'])['total_req_count'].idxmax()]

# Sorting the results for better readability
sorted_max_req_per_day = max_req_per_day.sort_values('total_req_count', ascending=False)

# Plotting the results for the selected service
plt.figure(figsize=(15, 7))
sns.barplot(x='hour', y='total_req_count', hue='day_of_week', data=grouped_data)
plt.title(f'Average Total Request Count by Day of Week and Hour for Service ID {selected_service_id}')
plt.xlabel('Hour of Day')
plt.ylabel('Average Total Request Count')
plt.xticks(np.arange(24))
plt.legend(title='Day of Week')
plt.grid(True)
plt.tight_layout()
plt.show()

# Constructing the result sentence with a 2-hour range
surges = []
for index, row in sorted_max_req_per_day.iterrows():
    day = row['day_of_week']
    hour = int(row['hour'])
    next_hour = (hour + 1) % 24  # Change to a 2-hour range
    avg_requests = int(row['total_req_count'])
    am_pm = "AM" if hour < 12 else "PM"
    next_am_pm = "AM" if next_hour < 12 or next_hour == 24 else "PM"  # Adjust for 24-hour format
    hour = hour if hour <= 12 else hour - 12
    next_hour = next_hour if next_hour <= 12 or next_hour == 24 else next_hour - 12  # Adjust for 12-hour format
    surge = f"On {day}s from {hour} {am_pm} to {next_hour} {next_am_pm}, with an average of approximately {avg_requests:,} requests"
    surges.append(surge)

# Combine the sentences with an 'and' before the last item
if len(surges) > 1:
    result_sentence = ", ".join(surges[:-1]) + ", and " + surges[-1]
else:
    result_sentence = surges[0]

print(f"The highest surges in volume for Service ID {selected_service_id} occur " + result_sentence + ".")