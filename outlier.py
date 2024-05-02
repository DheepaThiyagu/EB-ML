import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from appp import project_path,training_path
import os
# Assuming your DataFrame is 'df' with a column 'error_count'
# You may need to replace 'error_count' with the actual column name in your DataFrame
file_path = r'C:\Users\91701\EB_ML\training_data.csv'
df= pd.read_csv(file_path)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Contamination is the expected proportion of anomalies
model.fit(df[['error_count']])

# Predict outliers/anomalies
df['anomaly_score'] = model.decision_function(df[['error_count']])
df['is_anomaly'] = model.predict(df[['error_count']])

# Plot the results
plt.scatter(df.index, df['error_count'], c=df['is_anomaly'], cmap='viridis', marker='x')
plt.title('Isolation Forest Anomaly Detection for Error Count')
plt.xlabel('Index')
plt.ylabel('Error Count')
plt.show()

# Display the data with anomaly scores and labels
print(df[['error_count', 'anomaly_score', 'is_anomaly']])
