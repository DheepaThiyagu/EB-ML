from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
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

# Initialize an empty list to store breach volumes for each service ID
pred_breach_volumes = []
actual_volumes=[]

# Initialize lists to store actual and predicted labels
actual_labels = []
predicted_labels = []

# Iterate over unique service IDs
for sid in df_corr_vol_err['sid'].unique():
    # Filter data for the current service ID
    service_data = data[data['sid'] == sid]
    X = service_data[['record_time','sid','total_req_count', 'error_count']]
    y = service_data['is_eb_breached']

    # Check the number of samples for the current service ID
    if len(X) < 4:
        # If there is only one sample, consider its volume as breach volume
        print(f"Not enough data to train for {sid}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # # Predict probabilities on the test set
        # y_prob = model.predict_proba(X)
        # print("sid and y-prob",sid,y_prob)

        # Predict probabilities on the test set
        y_pred = model.predict(X_test)
        print("sid and y-pred",sid,y_pred)

        # Append actual and predicted labels to the lists
        actual_labels.extend(y_test)
        predicted_labels.extend(y_pred)

        # Select threshold based on desired certainty level
        threshold = 0.8  # Adjust as needed

        # Threshold volume above which breach is likely to happen
        pred_breach_volume = X_test[y_pred==1]['total_req_count'].min()  # Adjust threshold as needed

        actual_volume = X_test[y_test==1]['total_req_count'].min()

        # # Find the volume above which the predicted probability exceeds the threshold
        # if len(y_prob[0]) == 2:  # Ensure the prediction probabilities have two columns
        #     pred_breach_volume = X[y_prob[:, 1] >= threshold]['total_req_count'].min()
        # else:
        #     # If the prediction probabilities have only one column, consider all instances as positive class
        #     # Assuming positive class is represented by label 1
        #     positive_class_label = 1  # Adjust as needed based on your model's class labels
        #
        #     # Apply threshold to determine the predicted class
        #     predicted_class = (y_prob[:, 0] >= threshold).astype(int)
        #
        #     # Select the instances where the predicted class is the positive class
        #     positive_instances = X[predicted_class == positive_class_label]
        #
        #     # Find the minimum total_req_count among these instances
        #     pred_breach_volume = positive_instances['total_req_count'].min()

    # Append the breach volume for the current service ID to the list
    pred_breach_volumes.append({'sid': sid, 'breach_volume': pred_breach_volume})
    actual_volumes.append({'sid': sid, 'actual_volume': actual_volume})

# Calculate evaluation metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Convert the list of breach volumes to a DataFrame
breach_volumes_df = pd.DataFrame(pred_breach_volumes)
breach_volumes_df=breach_volumes_df.sort_values(by='sid', ascending=True)


# Convert the list of breach volumes to a DataFrame
actual_volumes_df = pd.DataFrame(actual_volumes)
actual_volumes_df=actual_volumes_df.sort_values(by='sid', ascending=True)

# Concatenate the DataFrames by 'sid'
# combined_df = pd.concat([breach_volumes_df, actual_volumes_df], axis=0, keys=['pred_breach_volumes', 'actual_volumes'])
# Merge the DataFrames on 'sid'
combined_df = pd.merge(breach_volumes_df, actual_volumes_df, on='sid', suffixes=('_pred', '_actual'))


# combined_df = pd.concat([breach_volumes_df,actual_volumes_df], ignore_index=True)


# # Define the lists
# sid_list = [sid for sid, _ in true_data]
# true_volume_list = [true_vol for _, true_vol in true_data]
# breach_volume_list = [breach_vol for sid, breach_vol in breach_data]
#
# # Create a DataFrame
# df = pd.DataFrame({
#     'sid': sid_list,
#     'true_volume': true_volume_list,
#     'breach_volume': breach_volume_list
# })

# Save the breach volumes to a CSV file
breach_volumes_file_path = 'Prob_alldata_rf_clssfr_brl_vol_per_service.csv'
combined_df.to_csv(breach_volumes_file_path, index=False)

print(f"Breach volumes saved to CSV file: {breach_volumes_file_path}")