from datetime import datetime, timedelta
from datetime import datetime, timedelta


# Set the current date and time
# current_date = datetime(2024, 1, 29, 1, 0, 0)
current_date=datetime.now()
# Set start time to 00:00 hour of the previous day
start_time = current_date - timedelta(days=1)
start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

# Set end time to 23:59 of the previous day
end_time = current_date - timedelta(days=1)
end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=0)

# Convert to human-readable format
current_date_str = current_date.strftime('%Y-%m-%d %H:%M:%S ')
start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S ')
end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

print("Current time:", current_date_str)
print("Start time:", start_time_str)
print("End time:", end_time_str)

# Convert to timestamps (in seconds) and then to milliseconds
current_timestamp_ms = int(current_date.timestamp() * 1000)
start_time_ms = int(start_time.timestamp() * 1000)
end_time_ms = int(end_time.timestamp() * 1000)

print("Current time:", current_timestamp_ms)
print("Training data_Start time:", start_time_ms)
print("Training data _End time:", end_time_ms)

# Calculate the prediction start date (10 days ahead)
prediction_start_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
prediction_end_date = prediction_start_date + timedelta(days=10) - timedelta(seconds=1)

# Convert the dates to strings
pred_start_date = prediction_start_date.strftime('%Y-%m-%d %H:%M:%S')
pred_end_date = prediction_end_date.strftime('%Y-%m-%d %H:%M:%S')

print("Prediction Start Date:", pred_start_date)
print("Prediction End Date:", pred_end_date)


tomorrow_date_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
print(tomorrow_date_start)