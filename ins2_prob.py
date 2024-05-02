from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dataset
file_path = r'C:\Users\91701\EB_ML\fivemins_wm_oseb1decv1_753.csv'
data = pd.read_csv(file_path)

# Initialize an empty list to store breach volumes for each service ID
breach_volumes = []

# # Features and target variable
X = data[['record_time','sid','total_req_count', 'error_count']]
y = data['is_eb_breached']
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize an empty list to store dictionaries
result_data = []

# Iterate over unique service IDs
for sid in data['sid'].unique():
    # Filter data for the current service ID
    service_data = data[data['sid'] == sid]
    # Check if there is more than one unique class for the current service ID
    # unique_classes = service_data['is_eb_breached'].unique()
    # if len(unique_classes) == 1:
    #     # If there is only one unique class, use it as the breach volume
    #     breach_volume = 0 if unique_classes[0] == 0 else service_data['total_req_count'].max()
    # else:
    X_service = service_data[['record_time','sid','total_req_count', 'error_count']]
    y_service = service_data['is_eb_breached']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_prob = model.predict_proba(X_service)[:, 1]


    # Calibration curve to assess model calibration
    prob_true, prob_pred = calibration_curve(y_service, y_prob, n_bins=10, strategy='uniform')
    # A calibration curve is a graphical representation of the relationship between the predicted
    # probabilities and the true probabilities or outcomes. By computing the calibration curve,
    # we can assess how well the predicted probabilities align with the actual outcomes, providing insights into the model's calibration performance.

    # Select threshold based on desired certainty level
    threshold = 0.8  # Adjust as needed

    # Find the volume range where the predicted probability exceeds the threshold
    high_certainty_volumes = X_service[y_prob >= threshold]['total_req_count']

    # Append results to the list
    result_data.append({'sid': sid,
                        'min_volume': high_certainty_volumes.min(),
                        'max_volume': high_certainty_volumes.max()})

# Convert the list of dictionaries to a DataFrame
result_df = pd.DataFrame(result_data)

# Save the results to a CSV file
result_file_path = 'high_certainty_volume_ranges_per_service.csv'
result_df.to_csv(result_file_path, index=False)

print(f"Results saved to CSV file: {result_file_path}")

# Predict on the test set
y_pred = model.predict(X_test)

# Get the "sid" column for the samples in X_test
test_sids = data.loc[X_test.index, 'sid']

# Concatenate X_test, y_test, and y_pred along with the "sid" column
test_results = pd.concat([test_sids.reset_index(drop=True), X_test.reset_index(drop=True), y_test.reset_index(drop=True), pd.Series(y_pred, name='predicted')], axis=1)

# Save the test results to a CSV file
test_results_file_path = 'test_results.csv'
test_results.to_csv(test_results_file_path, index=False)

print(f"Test results saved to CSV file: {test_results_file_path}")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
