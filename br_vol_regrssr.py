from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from appp import classify_services,list_unique_services
# Load the dataset
file_path = r'C:\Users\91701\EB_ML\training_data.csv'
data = pd.read_csv(file_path)

correlation_threshold = 0.70

vec_data, venc_data, max_correlations_vol_err = classify_services(data,correlation_threshold,'vol_err')
# vrc_data, vrnc_data, max_correlations_vol_resp = classify_services(data,correlation_threshold,'vol_resp')

print(vec_data,max_correlations_vol_err)
df_corr_vol_err = list_unique_services(vec_data, max_correlations_vol_err)
df_corr_vol_err=df_corr_vol_err.sort_values(by='sid')

corr_vol_err_file_path = 'corr_vol_err.csv'
df_corr_vol_err.to_csv(corr_vol_err_file_path, index=False)
print(f"Sid with Max correlationsb/w vol and err saved : {corr_vol_err_file_path}")


# Drop rows with missing values
data.dropna(inplace=True)

# Initialize an empty list to store breach volumes for each service ID
breach_data = []
true_data = []
data=data.sort_values(by='sid')
# Iterate over unique service IDs
for sid in df_corr_vol_err['sid'].unique():
    # Filter data for the current service ID
    service_data = data[data['sid'] == sid]

    # Features and target variable
    X = service_data[['record_time','total_req_count', 'error_count']]
    y = service_data['is_eb_breached']

    if len(service_data) <4:
        print(f"Not enough samples for {sid} for training. Please add more data.")
    else:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        #  # Combine X_test with y_test
        # test_data = X_test.copy()
        # test_data['y_test'] = y_test
        #
        # # Save the test data to a CSV file
        # test_data.to_csv(f'{sid}_test_data.csv')
        #
        # print(f"Test data saved to {sid}_test_data.csv'")


        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict breach volumes on the test set
        y_pred = model.predict(X)
        print("y-pred",sid,y_pred)

        # Threshold volume above which breach is likely to happen
        pred_threshold_volume = X[y_pred >= 0.8]['total_req_count'].min()  # Adjust threshold as needed

        true_volume = X[y== 1]['total_req_count'].min()

        print("thersold volume",sid,true_volume, pred_threshold_volume)

        # Append the breach volume, true volume, and service ID for the current service ID to the lists
        # if not np.isnan(threshold_volume):
        breach_data.append((sid, pred_threshold_volume))

        # True volume above which breach happened
        # if not np.isnan(true_volume):
        true_data.append((sid, true_volume))

# Define the lists
sid_list = [sid for sid, _ in true_data]
true_volume_list = [true_vol for _, true_vol in true_data]
breach_volume_list = [breach_vol for sid, breach_vol in breach_data]

# Create a DataFrame
df = pd.DataFrame({
    'sid': sid_list,
    'true_volume': true_volume_list,
    'breach_volume': breach_volume_list
})

df=df.sort_values(by='sid')
threshold_file_path = 'corr_sid_all_threshold_volumes_per_service.csv'
df.to_csv(threshold_file_path, index=False)
print(f"Threshold volumes saved to CSV file: {threshold_file_path}")

# Convert lists to numpy arrays
true_volume_array = np.array(true_volume_list)
breach_volume_array = np.array(breach_volume_list)

# Filter out NaN values from both arrays
valid_indices = ~np.isnan(true_volume_array) & ~np.isnan(breach_volume_array)
true_volume_clean = true_volume_array[valid_indices]
breach_volume_clean = breach_volume_array[valid_indices]


# Check if both lists have at least one element
if len(true_volume_clean) > 0 and len(breach_volume_clean) > 0:
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_volume_clean, breach_volume_clean)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(true_volume_clean, breach_volume_clean)

    # Calculate R-squared (R2) Score
    r2 = r2_score(true_volume_clean, breach_volume_clean)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
else:
    print("Both true_volume_list and breach_volume_list contain only NaN values. Cannot calculate error metrics.")


