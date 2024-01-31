import time

# from XGB_functions import train_model_with_grid_search,save_forecast_index, predict_future_dataset,plot_predictions, prediction_input,connect_db,add_lags,create_features,calculate_prediction_interval

# from XGB_functions import connectelk_retrive_data ,list_unique_services, classify_services,update_unique_services,connect_db,add_lags,create_features,calculate_prediction_interval,remove_outliers,train_test_splitdata,train_model,fit_model,predict_test,print_metrics,evaluate_metrics
from flask import Flask, render_template, request, jsonify
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import configparser
import argparse
from timeit import default_timer as timer
import certifi
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit,GridSearchCV
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,make_scorer
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler  # Example preprocessing step
from sqlalchemy import create_engine
from datetime import datetime
import argparse
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from ydata_profiling import ProfileReport
from concurrent.futures import ThreadPoolExecutor
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ap = Flask(__name__)

@ap.route('/')
def index():
    return 'Welcome to your EB_ML Flask App!'

project_path = os.getcwd()

# Specify the relative path to your models directory
models_path = 'models'
predictions_path='predictions'
training_path= 'training'
retraining_path='retraining'

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

elasticsearch_host = config.get('elasticsearch', 'host')
elasticsearch_port = config.getint('elasticsearch', 'port')
elk_username = config.get('elasticsearch', 'username')
elk_password = config.get('elasticsearch', 'password')


app_id="753"
source_index='fivemins_wm_oseb1decv1_753'
target_index='predictions_9_jan_by_script'
training_index='fivemins_753'
def train_model_with_grid_search(df):
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes = []
        mapes = []
        mases=[]

        for train_idx, val_idx in tss.split(df):
            req_vol_preds = []
            resp_time_preds = []
            err_cnt_preds = []

            train = df.iloc[train_idx]
            test= df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
            REQ_VOL_TARGET = 'total_req_count'
            RESP_TIME_TARGET = 'resp_time_sum'
            ERR_CNT_TARGET = 'error_count'

            X_train = train[FEATURES]
            y_train_req_vol = train[REQ_VOL_TARGET]
            y_train_resp_time = train[RESP_TIME_TARGET]
            y_train_err_cnt = train[ERR_CNT_TARGET]

            X_test = test[FEATURES]
            y_test_req_vol = test[REQ_VOL_TARGET]
            y_test_resp_time = test[RESP_TIME_TARGET]
            y_test_err_cnt = test[ERR_CNT_TARGET]

            # Define the hyperparameter grid for GridSearchCV

            param_grid = {
                'n_estimators': [500,800,1000],
                'learning_rate': [0.01, 0.2, 0.3],
                'max_depth': [13,15,20],
                # Add other hyperparameters to tune
            }
            # Create XGBoost regressor models
            reg_req_vol = xgb.XGBRegressor(base_score=0.5, booster='gbtree',objective='reg:squarederror')
            reg_resp_time = xgb.XGBRegressor(base_score=0.5, booster='gbtree',objective='reg:squarederror')
            reg_err_cnt = xgb.XGBRegressor(base_score=0.5, booster='gbtree',objective='reg:squarederror')

            # Create a scorer for GridSearchCV using negative mean squared error
            scorer = make_scorer(mean_squared_error, greater_is_better=False)

            # Create GridSearchCV instances for each regressor
            grid_search_req_vol = GridSearchCV(reg_req_vol, param_grid, scoring=scorer, cv=tss)
            grid_search_resp_time = GridSearchCV(reg_resp_time, param_grid, scoring=scorer, cv=tss)
            grid_search_err_cnt = GridSearchCV(reg_err_cnt, param_grid, scoring=scorer, cv=tss)

            # Fit GridSearchCV to the training data
            grid_search_req_vol.fit(X_train, y_train_req_vol.astype(int))
            grid_search_resp_time.fit(X_train, y_train_resp_time)
            grid_search_err_cnt.fit(X_train, y_train_err_cnt.astype(int))

            # Get the best hyperparameters
            best_params_req_vol = grid_search_req_vol.best_params_
            best_params_resp_time = grid_search_resp_time.best_params_
            best_params_err_cnt = grid_search_err_cnt.best_params_
            print("best parameters for volume, response time and error count",best_params_req_vol,best_params_resp_time,best_params_err_cnt)

            # Train the final models with the best hyperparameters
            reg_req_vol = xgb.XGBRegressor(objective='reg:squarederror', **best_params_req_vol)
            reg_resp_time = xgb.XGBRegressor(objective='reg:squarederror', **best_params_resp_time)
            reg_err_cnt = xgb.XGBRegressor(objective='reg:squarederror', **best_params_err_cnt)

            fit_model(X_train,y_train_req_vol,y_train_resp_time,y_train_err_cnt,X_test,y_test_req_vol,y_test_resp_time,y_test_err_cnt,reg_req_vol,reg_resp_time,reg_err_cnt)
            req_vol_pred,resp_time_pred,err_cnt_pred,req_vol_interval,resp_time_interval,err_cnt_interval= predict_test(X_test,FEATURES,reg_req_vol,reg_resp_time,reg_err_cnt)

            # Append intervals to your lists
            req_vol_preds.append((req_vol_pred, req_vol_interval))
            resp_time_preds.append((resp_time_pred, resp_time_interval))
            err_cnt_preds.append((err_cnt_pred, err_cnt_interval))

            # Create a DataFrame to store predictions and intervals
            predictions_df = pd.DataFrame({
                'Datetime': X_test.index,
                'Req_Vol_Pred': req_vol_pred,
                'Resp_Time_Pred': resp_time_pred,
                'Err_Cnt_Pred': err_cnt_pred,
                'Req_Vol_Lower': req_vol_interval[0],
                'Req_Vol_Upper': req_vol_interval[1],
                'Req_Vol_Confidence_Score':req_vol_interval[2],
                'Resp_Time_Lower': resp_time_interval[0],
                'Resp_Time_Upper': resp_time_interval[1],
                'Resp_Time_Confidence_Score':resp_time_interval[2],
                'Err_Cnt_Lower': err_cnt_interval[0],
                'Err_Cnt_Upper': err_cnt_interval[1],
                'Err_cnt_Confidence_Score':err_cnt_interval[2]
            })


            # Save the DataFrame to a CSV file
            predictions_df.to_csv('range_predictions.csv', index=False)


            req_vol_rmse,resp_time_rmse,err_cnt_rmse,req_vol_mae,resp_time_mae,err_cnt_mae,req_vol_mape,resp_time_mape,err_cnt_mape,req_vol_mase,resp_time_mase,err_cnt_mase=evaluate_metrics(y_test_req_vol, req_vol_pred,y_test_resp_time, resp_time_pred,y_test_err_cnt, err_cnt_pred)

            rmses.append((req_vol_rmse, resp_time_rmse,err_cnt_rmse))
            maes.append((req_vol_mae,resp_time_mae,err_cnt_mae))
            mapes.append((req_vol_mape,resp_time_mape,err_cnt_mape))
            mases.append((req_vol_mase,resp_time_mase,err_cnt_mase))
            return reg_req_vol,reg_resp_time,reg_err_cnt,rmses,maes,mapes,mases

    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service due to error: {e}")

        # Return the final models
    return None,None,None,None,None,None,None


def predict_future_dataset(app_id,model_req_vol,model_resp_time,model_err_cnt,sid,freq,start_date,end_date,filtered_df):
    # req_vol_preds = []
    # resp_time_preds = []
    # err_cnt_preds=[]
    # To ignore specific warnings
    print(app_id)
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')


    # df = filtered_df
    # print(df.head())
    #
    # df = df.set_index('datetime')
    # df.index = pd.to_datetime(df.index, format="%b %d, %Y @ %H:%M:%S.%f")

    # df.index = pd.to_datetime(df.index,format="%Y-%m-%d %H:%M:%S")


    pred_vol = xgb.XGBRegressor()
    pred_err=xgb.XGBRegressor()
    pred_resp=xgb.XGBRegressor()



    pred_vol.load_model(model_req_vol)
    pred_err.load_model(model_err_cnt)
    pred_resp.load_model(model_resp_time)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
    # 'lag1','lag2','lag3']
    # REQ_VOL_TARGET = 'req_vol'
    # RESP_TIME_TARGET = 'resp_time'
    # ERR_CNT_TARGET = 'err_cnt'

    # Create future dataframe
    future = pd.date_range(start_date,end_date, freq=freq) # freq is set for data granularity ( 5 mins or 1 hour)
    future_df = pd.DataFrame(index=future)

    future_w_features=create_features(future_df)
    # Convert app_id to a pandas Series
    app_id_series = pd.Series(app_id)

    future_w_features['datetime']=future_w_features.index
    future_w_features['app_id']=app_id_series.iloc[0]

    future_w_features['sid']=sid
    future_w_features['pred_req_vol'] = pred_vol.predict(future_w_features[FEATURES])
    future_w_features['pred_err_cnt'] = pred_err.predict(future_w_features[FEATURES])
    future_w_features['pred_resp_time'] = pred_resp.predict(future_w_features[FEATURES])

    future_w_features['pred_req_vol']=future_w_features['pred_req_vol'].astype(int)
    # future_w_features['pred_resp_time']=future_w_features['pred_resp_time'].astype(int)
    future_w_features['pred_err_cnt'] = future_w_features['pred_err_cnt'].astype(int)


    # Calculate prediction intervals
    req_vol_interval = calculate_prediction_interval(future_w_features['pred_req_vol'])
    resp_time_interval = calculate_prediction_interval(future_w_features['pred_resp_time'])
    err_cnt_interval = calculate_prediction_interval(future_w_features['pred_err_cnt'])

    future_w_features['Req_Vol_Lower']=req_vol_interval[0].astype(int)
    future_w_features['Req_Vol_Upper']= req_vol_interval[1].astype(int)
    future_w_features['Req_Vol_Conf_score']=req_vol_interval[2].astype(int)

    future_w_features['Resp_Time_Lower']= resp_time_interval[0]
    future_w_features['Resp_Time_Upper']= resp_time_interval[1]
    future_w_features['Resp_Time_Conf_score']=resp_time_interval[2]

    future_w_features['Err_Cnt_Lower']= err_cnt_interval[0]
    future_w_features['Err_Cnt_Upper']= err_cnt_interval[1]
    future_w_features['Err_Cnt_Conf_score']=err_cnt_interval[2]

    future_w_features['Timestamp'] = future_w_features['datetime'].astype('int64') // 10**6

    # # Concatenate actual and predicted dataframes based on datetime
    # if filtered_df is not None:
    #     print(filtered_df.head())
    #     # Convert 'record_time' to milliseconds in the 'actual_df'
    #     filtered_df['Timestamp'] = filtered_df.index.astype('int64') // 10**6
    #     filtered_df['sid'] = filtered_df['sid'].astype(str)
    future_w_features['sid']=future_w_features['sid'].astype(str)
    #
    #     # Merge the two DataFrames on 'record_time', 'app_id', and 'sid'
    #     merged_df = pd.merge(future_w_features, filtered_df, how='outer', on=['Timestamp', 'app_id', 'sid'])
    # else:
    #     merged_df = future_w_features
    #
    # # Print or use the merged DataFrame as needed
    # print(merged_df)

    # plot_predictions(filtered_df,future_w_features,sid,freq)
    print("returing the predicted values")
    return future_w_features
    # return merged_df
#

# def predict_future_dataset(app_id, model_req_vol, model_resp_time, model_err_cnt, sid, freq, start_date, end_date, filtered_df):
#     # To ignore specific warnings
#     warnings.filterwarnings("ignore", category=FutureWarning)
#
#     # Create future dataframe
#     future = pd.date_range(start_date, end_date, freq=freq)  # freq is set for data granularity (5 mins or 1 hour)
#     future_df = pd.DataFrame(index=future)
#
#     # Convert app_id to a pandas Series
#     app_id_series = pd.Series(app_id)
#
#     future_df['datetime'] = future_df.index
#     future_df['app_id'] = app_id_series.iloc[0]
#     future_df['sid'] = sid
#
#     # Load models if available
#     pred_vol, pred_resp, pred_err = None, None, None
#     if model_req_vol:
#         pred_vol = xgb.XGBRegressor()
#         pred_vol.load_model(model_req_vol)
#
#     if model_resp_time:
#         pred_resp = xgb.XGBRegressor()
#         pred_resp.load_model(model_resp_time)
#
#     if model_err_cnt:
#         pred_err = xgb.XGBRegressor()
#         pred_err.load_model(model_err_cnt)
#
#     FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
#
#     # Predict future data
#     if pred_vol:
#         future_df['pred_req_vol'] = pred_vol.predict(create_features(future_df)[FEATURES]).astype(int)
#     else:
#         future_df['pred_req_vol'] = None
#
#     if pred_resp:
#         future_df['pred_resp_time'] = pred_resp.predict(create_features(future_df)[FEATURES])
#     else:
#         future_df['pred_resp_time'] = None
#
#     if pred_err:
#         future_df['pred_err_cnt'] = pred_err.predict(create_features(future_df)[FEATURES]).astype(int)
#     else:
#         future_df['pred_err_cnt'] = None
#
#     # Calculate prediction intervals
#     future_df = calculate_prediction_intervals(future_df)
#
#     # Merge with actual data (if available)
#     if filtered_df is not None:
#         merged_df = pd.merge(future_df, filtered_df, how='outer', on=['datetime', 'app_id', 'sid'])
#     else:
#         merged_df = future_df
#
#     # Print or use the merged DataFrame as needed
#     print(merged_df)
#
#     return merged_df

# def calculate_prediction_intervals(df):
#     # Calculate prediction intervals for each predicted column
#     for target in ['req_vol', 'resp_time', 'err_cnt']:
#         lower_col = f'{target}_lower'
#         upper_col = f'{target}_upper'
#         conf_score_col = f'{target}_conf_score'
#
#         if f'pred_{target}' in df.columns:
#             df[lower_col], df[upper_col], df[conf_score_col] = calculate_prediction_interval(df[f'pred_{target}'])
#
#     return df

# def calculate_prediction_interval(predictions):
#     # Custom logic to calculate prediction intervals and confidence scores
#     # Replace this with your own implementation based on your requirements
#     print("inside calculating interval")
#     lower_bound = predictions - 0.1 * predictions
#     upper_bound = predictions + 0.1 * predictions
#     confidence_score = 0.9
#
#     return lower_bound, upper_bound, confidence_score

def bulk_save_training_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password):
    def get_headers():
        headers = {}
        if elk_username and elk_password:
            headers['Authorization'] = 'Basic ' + get_secret()
        return headers

    def get_auth():
        if elk_username and elk_password:
            return HTTPBasicAuth(elk_username, elk_password)
        else:
            return None

    def get_secret():
        return elk_username + ':' + elk_password

    def push_bulk_data_to_elk(bulkMsg):
        endpoint = f'{elasticsearch_host}:{elasticsearch_port}/_bulk'
        response = requests.post(endpoint, data=bulkMsg, auth=get_auth(), headers={'Content-Type': 'application/json'}, verify=False)
        print(bulkMsg)
        # Check for errors
        if response.status_code == 200:
            print(f"Documents indexed successfully.")
        else:
            print(f"Error indexing documents: {response.status_code}, {response.text}")
    # Convert DataFrame to JSON with Timestamp objects as strings
    df_json = df.to_json(orient='records', date_format='iso')

    # Push data using bulk index
    data_list = json.loads(df_json)

    # Push data using bulk indexing
    bulk_msg = '\n'.join([
        # f'{{"index": {{"_index": "{target_index}"}}}}\n{json.dumps({**row, "record_time": int(row["record_time"].timestamp() * 1000)})}'
          f'{{"index": {{"_index": "{target_index}"}}}}\n{json.dumps(row)}'
        for row in data_list
    ]) + '\n'
    push_bulk_data_to_elk(bulk_msg)


    # Refresh the index to make the documents available for search
    requests.post(f'{elasticsearch_host}:{elasticsearch_port}/{target_index}/_refresh', auth=get_auth(), verify=False)

    print(f"Training dataframe successfully pushed to Elasticsearch index: {target_index}")
def bulk_save_forecast_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password):
    def get_headers():
        headers = {}
        if elk_username and elk_password:
            headers['Authorization'] = 'Basic ' + get_secret()
        return headers

    def get_auth():
        if elk_username and elk_password:
            return HTTPBasicAuth(elk_username, elk_password)
        else:
            return None

    def get_secret():
        return elk_username + ':' + elk_password

    def push_bulk_data_to_elk(bulkMsg):
        endpoint = f'{elasticsearch_host}:{elasticsearch_port}/_bulk'
        response = requests.post(endpoint, data=bulkMsg, auth=get_auth(), headers={'Content-Type': 'application/json'}, verify=False)
        print(bulkMsg)
        # Check for errors
        if response.status_code == 200:
            print(f"Documents indexed successfully.")
        else:
            print(f"Error indexing documents: {response.status_code}, {response.text}")
    try:
        # Convert DataFrame to JSON with Timestamp objects as strings
        df_json = df.to_json(orient='records', date_format='iso')

        # Push data using bulk index
        data_list = json.loads(df_json)

        # Push data using bulk indexing
        bulk_msg = '\n'.join([
            f'{{"index": {{"_index": "{target_index}"}}}}\n{json.dumps({**row, "record_time": int(pd.to_datetime(datetime.now()).timestamp() * 1000)})}'
            # f'{{"index": {{"_index": "{target_index}"}}}}\n{json.dumps(row)}'
            for row in data_list
        ]) + '\n'
        push_bulk_data_to_elk(bulk_msg)


        # Refresh the index to make the documents available for search
        requests.post(f'{elasticsearch_host}:{elasticsearch_port}/{target_index}/_refresh', auth=get_auth(), verify=False)

        print(f"Prediction dataframe successfully pushed to OPensearch index: {target_index}")
    except Exception as e:
        # Handle the exception (e.g., print an error message)
        print(f"Error pushing data to OpenSearch index: {e}")

# def save_forecast_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password):
#
#     def get_headers():
#         headers = {}
#         if elk_username and elk_password:
#             headers['Authorization'] = 'Basic ' + get_secret()
#
#         print("Headers:", headers)
#         return headers
#     def get_auth():
#         if elk_username and elk_password:
#             return HTTPBasicAuth(elk_username, elk_password)
#         else:
#             return None
#     def get_secret():
#         return elk_username + ':' + elk_password
#     def push_bulk_data_to_elk(bulkMsg):
#         # print(endpoint)
#         print(bulkMsg)
#         # Elasticsearch endpoint
#         endpoint = f'{elasticsearch_host}:{elasticsearch_port}/_bulk'
#         response = requests.post(endpoint, data=bulkMsg, auth=get_auth(), headers={'Content-Type': 'application/json'}, verify=False)
#
#         # Check for errors
#         if response.status_code == 200:
#             print(f"Document indexed successfully till: {index + 1}.")
#         else:
#             print(f"Error indexing document {index + 1}: {response.status_code}, {response.text}")
#
#
#     # elasticsearch_host = 'https://ec2-54-82-37-97.compute-1.amazonaws.com'
#     # elasticsearch_port = 9200
#     # target_index = 'test_ai_dheepa_13dec_01'
#     # elk_username = 'admin'
#     # elk_password = 'admin'
#
#     iteration = 0
#     # df = df.fillna(0)  # Replace NaN with the string 'null'
#
#     # rowNum = 0
#     bulk_msg = ""
#     # Iterate through rows in the DataFrame and push each row as a document to Elasticsearch
#     for index, row in df.iterrows():
#         # rowNum += 1
#         document = row.to_dict()
#         if len(bulk_msg) == 0:
#             bulk_msg += '{"index":{"_index":"' + target_index + '"}}\n'
#
#         bulk_msg += json.dumps(document, default=str) + '\n'
#
#         if len(bulk_msg) >= 500:
#             push_bulk_data_to_elk(bulk_msg)
#             bulk_msg = ""
#     #     if rowNum%500 == 0:
#     #         push_bulk_data_to_elk(bulkMsg)
#     #         bulkMsg = ""
#     #     else:
#     #         bulkMsg = bulkMsg + '{"index":{"_index":"'+target_index+'"}'
#     #         bulkMsg = bulkMsg + '\n'
#     #         bulkMsg = bulkMsg + json.dumps(document)
#     #         bulkMsg = bulkMsg + '\n'
#     #
#     #
#     # if rowNum%500 != 0:
#     if bulk_msg:
#         push_bulk_data_to_elk(bulk_msg)
#
#
#     # Refresh the index to make the documents available for search
#     requests.post(f'{elasticsearch_host}:{elasticsearch_port}/{target_index}/_refresh', auth=get_auth(), verify=False)
#
#     print(f"Prediction dataframe successfully pushed to Elasticsearch index: {target_index}")




def connectelk_retrive_data(app_id, start_time, end_time,source_index):

    def check_elk_connection(url_string):
        try:
            # Logging URL
            print(f"url_string: {url_string}")

            # response = requests.get(url_string, headers=get_headers(), auth=get_auth(), verify=certifi.where())
            response = requests.get(url_string, auth=get_auth(), verify=False)

            response.raise_for_status()  # Raise an error for 4xx and 5xx status codes

            if response.status_code == requests.codes.ok:
                return True  # Connection successful
            else:
                return False

        except requests.exceptions.RequestException as e:
            # Handle exceptions
            print(f"Error making request: {e}")
            return False

    def get_headers():
        headers = {}
        if elk_username and elk_password:
            headers['Authorization'] = 'Basic ' + get_secret()

        print("Headers:", headers)
        return headers

    def get_auth():
        if elk_username and elk_password:
            return HTTPBasicAuth(elk_username, elk_password)
        else:
            return None

    def get_secret():
        return elk_username + ':' + elk_password
    def get_next_records_with_scroll_id(elk_host_url, scroll_id, orig_df):
        while(True):
            try:
                es_query = {
                    "scroll": "1m",
                    "scroll_id": scroll_id
                }
                response = requests.post(elk_host_url+'/scroll', auth=get_auth(), json=es_query, verify=False)
                response.raise_for_status()
                data = response.json()
                scroll_id = data.get('_scroll_id')
                hits = data.get('hits', {}).get('hits', [])

                if not hits:
                    return orig_df

                documents = [hit.get('_source', {}) for hit in hits]
                result_df = pd.DataFrame(documents)
                orig_df = orig_df._append(result_df, ignore_index=True)
                print("_scroll_id:", scroll_id)
                print("length of orig_df:",len(orig_df))
                # return original_df
            except requests.exceptions.RequestException as e:
                print(f"Request failed with error: {e}")

    # elk_host_url = 'https://ec2-54-82-37-97.compute-1.amazonaws.com:9200/_search'
    # url_string = 'https://ec2-54-82-37-97.compute-1.amazonaws.com:9200/daily_wm_oseb1decv1_753/_search'
    # elk_username = 'admin'
    # elk_password = 'admin'

    elk_host_url = f'{elasticsearch_host}:{elasticsearch_port}/_search'
    url_string = f'{elasticsearch_host}:{elasticsearch_port}/{source_index}/_search'

    result = check_elk_connection(url_string)
    print(f"Request result: {result}")

    # Your Elasticsearch query
    es_query = {
        "size": 500,
        "query": {
            "bool": {
                "filter": [
                    {"range": {"record_time": {"from":   start_time, "to": end_time, "include_lower": True, "include_upper": True, "boost": 1.0}}},
                    {"term": {"sub_datatype": {"value": "SERVICE", "boost": 1.0}}},
                    {"term": {"app_id": {"value": app_id, "boost": 1.0}}},
                    {"range": {"scripted_metric.total_count": {"from": 0, "to": None, "include_lower": False, "include_upper": True, "boost": 1.0}}}
                ],
                "adjust_pure_negative": True,
                "boost": 1.0
            }
        }
    }

    try:
        # Make the POST request to Elasticsearch
        response = requests.post(url_string+'?scroll=2m', auth=get_auth(), json=es_query, verify=False)

        # Check if the request was successful (status code 200)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Extract relevant data from the response (customize this based on your Elasticsearch document structure)
        hits = data.get('hits', {}).get('hits', [])
        documents = [hit.get('_source', {}) for hit in hits]

        # Create a DataFrame
        df = pd.DataFrame(documents)

        # fetch next records till last
        scroll_id = data.get('_scroll_id')
        print("_scroll_id:", scroll_id)

        df = get_next_records_with_scroll_id(elk_host_url, scroll_id, df)
        print("Total records in df:", len(df))

        # Display the columns and the DataFrame
        print("Columns:", df.columns)
        print("DataFrame:")
        print(df)

        # # Save the DataFrame to a CSV file
        #
        # csv_file_path = 'output_dataframe.csv'  # Provide the desired file path
        # df.to_csv(csv_file_path, index=False)
        # print(f"DataFrame saved to: {csv_file_path}")

        # Extract relevant fields from the "scripted_metric" column
        df_extracted = df['scripted_metric'].apply(pd.Series)

        # Select specific columns from the original DataFrame
        selected_columns = ['total_req_count', 'error_count', 'resp_time_sum','is_eb_breached','is_response_breached']
        df_selected = df[['record_time', 'app_id', 'sid']].join(df_extracted[selected_columns])

        # Display the resulting DataFrame
        print("Selected DataFrame:")
        print(df_selected)

        # Save the selected DataFrame to a CSV file

        # csv_file_path_selected = os.path.join(project_path, training_path,f'{source_index}.csv')
        # df_selected.to_csv(csv_file_path_selected, index=False)
        # print(f"Selected DataFrame saved to: {csv_file_path_selected}")
        return df_selected

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            print("Error 404: Resource Not Found. Handle this case appropriately.")
            # Perform actions or return a specific value when the resource is not found
            return None
        else:
            print(f"HTTP error occurred: {http_err}")
            # Handle other HTTP errors as needed
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed with error: {e}")



def plot_predictions(df,future_w_features,service_name,freq):
    # Create a figure and subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    # Set the main title for the entire figure
    fig.suptitle(f'Predictions for {service_name} ({freq})', fontsize=16)


    # Plot Request Volume
    axes[0].plot(df.index, df['total_req_count'], label='Actual Request Volume', color='red', marker='.', markersize=2, linewidth=1)
    axes[0].plot(future_w_features.index, future_w_features['pred_req_vol'], label='Predicted Request Volume', color='blue', marker='.', markersize=2, linewidth=1)
    axes[0].fill_between(future_w_features.index, future_w_features['Req_Vol_Lower'], future_w_features['Req_Vol_Upper'], color='blue', alpha=0.2)
    axes[0].set_ylabel('Request Volume')
    axes[0].legend()

    # Plot Response Time
    axes[1].plot(df.index, df['resp_time_sum'], label='Actual Response Time', color='orange', marker='.', markersize=2, linewidth=1)
    axes[1].plot(future_w_features.index, future_w_features['pred_resp_time'], label='Predicted Response Time', color='green', marker='o', markersize=2, linewidth=1)
    axes[1].fill_between(future_w_features.index, future_w_features['Resp_Time_Lower'], future_w_features['Resp_Time_Upper'], color='grey', alpha=0.2)
    axes[1].set_ylabel('Response Time')
    axes[1].legend()

    # Plot Error Count
    axes[2].plot(df.index, df['error_count'], label='Actual Error Count', color='aqua', marker='.', markersize=2, linewidth=1)
    axes[2].plot(future_w_features.index, future_w_features['pred_err_cnt'], label='Predicted Error Count', color='orange', marker='.', markersize=2, linewidth=1)
    axes[2].fill_between(future_w_features.index, future_w_features['Err_Cnt_Lower'], future_w_features['Err_Cnt_Upper'], color='green', alpha=0.2)
    axes[2].set_xlabel('Datetime')
    axes[2].set_ylabel('Error Count')
    axes[2].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()
    # canvas = FigureCanvas(fig)
    # png_output = io.BytesIO()
    # canvas.print_png(png_output)
    # response = make_response(png_output.getvalue())
    # response.headers['Content-Type'] = 'image/png'
    # plt.close(fig)  # Close the figure to free resources
    #





def update_unique_services(df):
    # Assuming that the DataFrame has a column named 'sid'
    unique_services = df['sid'].value_counts().index.tolist()

    # Create a new DataFrame with sid and record_count
    df_unique_services = pd.DataFrame({'sid': unique_services, 'record_count': df['sid'].value_counts().values})

    # Sort the DataFrame by record_count in descending order
    df_unique_services = df_unique_services.sort_values(by='record_count', ascending=False)

    # Display the head of the DataFrame
    print(df_unique_services.head())

    return df_unique_services


def load_input_model(sid=None):


    # service_name = service_name.replace('/', '_')
    if sid is not None:
        model_req_vol=os.path.join(project_path, models_path,f'{sid}_req_vol.json')
        model_resp_time=os.path.join(project_path, models_path,f'{sid}_resp_time.json')
        model_err_cnt=os.path.join(project_path, models_path,f'{sid}_err_cnt.json')
        # Check if JSON files exist before returning
        if all(os.path.exists(model) for model in [model_req_vol, model_resp_time, model_err_cnt]):
            return model_req_vol, model_resp_time, model_err_cnt
        else:
            # Handle the case where one or more files do not exist
            print(f"Error: One or more JSON files for {sid} do not exist.")
            return None,None,None
    else:
        model_req_vol=os.path.join(project_path, models_path,'model_req_vol.json')
        model_resp_time=os.path.join(project_path, models_path,'model_resp_time.json')
        model_err_cnt=os.path.join(project_path, models_path,'model_err_cnt.json')
        # Check if JSON files exist before returning
        if all(os.path.exists(model) for model in [model_req_vol, model_resp_time, model_err_cnt]):
            return model_req_vol, model_resp_time, model_err_cnt
        else:
            # Handle the case where one or more files do not exist
            print(f"Error: One or more JSON files  do not exist.")
            return None,None,None

def connect_db():

    db_username = 'watermelon'
    db_password = 'watermelon123'
    db_host = 'ec2-13-215-184-217.appp-southeast-1.compute.amazonaws.com'
    db_port = '30003'
    db_name = 'error_budget_ai'

    # Create a connection to the PostgreSQL database
    engine = create_engine(f'postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
    # try:
    # # Try to connect to the database
    #     connection = engine.connect()
    #     print("Connected to the database.")
    # except Exception as e:
    #     print(f"Error connecting to the database: {e}")
    # finally:
    # # Close the connection, if it was established
    #     if 'connection' in locals():
    #         connection.close()
    return engine


def create_features(df):

    # Add these components as features to your DataFrame
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week


    # df['lag1_err_cnt'] = df['error_count'].shift(1)
    # df['lag1_vol'] = df['total_req_count'].shift(1)
    # df['lag1_resp_time'] = df['resp_time_sum'].shift(1)
    # # Add rolling statistics
    # df['rolling_mean_vol'] = df['total_req_count'].rolling(window=3).mean()
    # df['rolling_std_err_cnt'] = df['error_count'].rolling(window=3).std()
    #
    # # Add interaction terms
    # df['vol_err_interaction'] = df['total_req_count'] * df['error_count']
    return df
def add_lags(df):
    target_map = df['total_req_count'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('40 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('50 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('60 days')).map(target_map)
    return df

def calculate_prediction_interval(predictions, alpha=0.99):
    lower_bound = np.percentile(predictions, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(predictions, (1 + alpha) / 2 * 100)
    # Calculate confidence score as the width of the interval
    confidence_score = upper_bound - lower_bound
    return lower_bound, upper_bound,confidence_score


def remove_outliers(df):
    # Remove outliers for req_vol
    Q1_req_vol = df['total_req_count'].quantile(0.15)
    Q3_req_vol = df['total_req_count'].quantile(0.95)
    IQR_req_vol = Q3_req_vol - Q1_req_vol

    outlier_mask_req_vol = (df['total_req_count'] < (Q1_req_vol - 1.5 * IQR_req_vol)) | (df['total_req_count'] > (Q3_req_vol + 1.5 * IQR_req_vol))
    df = df[~outlier_mask_req_vol]

    # Remove outliers for resp_time
    Q1_resp_time = df['resp_time_sum'].quantile(0.15)
    Q3_resp_time = df['resp_time_sum'].quantile(0.95)
    IQR_resp_time = Q3_resp_time - Q1_resp_time

    outlier_mask_resp_time = (df['resp_time_sum'] < (Q1_resp_time - 1.5 * IQR_resp_time)) | (df['resp_time_sum'] > (Q3_resp_time + 1.5 * IQR_resp_time))
    df = df[~outlier_mask_resp_time]

    # Remove outliers for err_cnt
    Q1_err_cnt = df['error_count'].quantile(0.15)
    Q3_err_cnt = df['error_count'].quantile(0.95)
    IQR_err_cnt = Q3_err_cnt - Q1_err_cnt

    outlier_mask_err_cnt = (df['error_count'] < (Q1_err_cnt - 1.5 * IQR_err_cnt)) | (df['error_count'] > (Q3_err_cnt + 1.5 * IQR_err_cnt))
    df = df[~outlier_mask_err_cnt]
    return df


def train_test_splitdata(df):
    total_records = len(df)
    print("Total records",total_records)
    split_percentage = 0.7  # 70% for training set

    split_index = int(total_records * split_percentage)
    print("split index",split_index)
    sorted_dates = df.index.sort_values()
    print(sorted_dates)
    split_date = sorted_dates[split_index]
    # Split the dataset into training and test sets
    print("Split date",split_date)
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    #
    # fig, ax = plt.subplots(figsize=(15, 5))
    #
    # # Plot volume from the training set
    # train['total_req_count'].plot(ax=ax, label='Training Set - Volume')
    #
    # # Plot volume from the test set
    # test['total_req_count'].plot(ax=ax, label='Test Set - Volume')
    #
    # # Plot response time from the training set
    # train['resp_time_sum'].plot(ax=ax, label='Training Set - Response Time')
    #
    # # Plot response time from the test set
    # test['resp_time_sum'].plot(ax=ax, label='Test Set - Response Time')
    #
    # # Plot response time from the training set
    # train['error_count'].plot(ax=ax, label='Training Set - Error count')
    #
    # # Plot response time from the test set
    # test['error_count'].plot(ax=ax, label='Test Set - Error count')
    #
    # ax.axvline(split_date, color='black', ls='--')
    # ax.legend()
    # plt.title('Volume and Response Time and Error in Training and Test Sets')
    # plt.show()
    return train,test,df
def print_metrics(sid,values,metrics):
    print(f'{metrics} for {sid}: ')
    print(f'Score across folds (Request Volume): {np.mean([score[0] for score in values]):0.4f}')
    print(f'Score across folds (Response Time): {np.mean([score[1] for score in values]):0.4f}')
    print(f'Score across folds (Error count): {np.mean([score[2] for score in values]):0.4f}')


    print(f'Fold scores (Request Volume): {[score[0] for score in values]}')
    print(f'Fold scores (Response Time): {[score[1] for score in values]}')
    print(f'Fold scores (Error count): {[score[2] for score in values]}')


def fit_model(X_train,y_train_req_vol,y_train_resp_time,y_train_err_cnt,X_test,y_test_req_vol,y_test_resp_time,y_test_err_cnt,reg_req_vol,reg_resp_time,reg_err_cnt):
    reg_req_vol.fit(X_train, y_train_req_vol,
                    eval_set=[(X_train, y_train_req_vol), (X_test, y_test_req_vol)],
                    verbose=100)

    reg_resp_time.fit(X_train, y_train_resp_time,
                      eval_set=[(X_train, y_train_resp_time), (X_test, y_test_resp_time)],
                      verbose=100)

    reg_err_cnt.fit(X_train, y_train_err_cnt,
                    eval_set=[(X_train, y_train_err_cnt), (X_test, y_test_err_cnt)],
                    verbose=100)

def mase(y_true, y_pred):
    n = len(y_true)

    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Calculate mean absolute difference between consecutive actual values
    denom = np.mean(np.abs(np.diff(y_true)))

    # Exclude instances where the actual value is zero
    non_zero_indices = y_true != 0
    abs_errors_non_zero = abs_errors[non_zero_indices]

    # Calculate MASE
    mase_value = np.mean(abs_errors_non_zero) / denom

    return mase_value

def evaluate_metrics(y_test_req_vol, req_vol_pred,y_test_resp_time, resp_time_pred,y_test_err_cnt, err_cnt_pred):
    req_vol_rmse = np.sqrt(mean_squared_error(y_test_req_vol, req_vol_pred))
    resp_time_rmse = np.sqrt(mean_squared_error(y_test_resp_time, resp_time_pred))
    err_cnt_rmse = np.sqrt(mean_squared_error(y_test_err_cnt, err_cnt_pred))

    req_vol_mae= mean_absolute_error(y_test_req_vol, req_vol_pred)
    resp_time_mae = mean_absolute_error(y_test_resp_time, resp_time_pred)
    err_cnt_mae = mean_absolute_error(y_test_err_cnt, err_cnt_pred)

    req_vol_mape = mean_absolute_percentage_error(y_test_req_vol, req_vol_pred)
    resp_time_mape = mean_absolute_percentage_error(y_test_resp_time, resp_time_pred)
    err_cnt_mape = mean_absolute_percentage_error(y_test_err_cnt, err_cnt_pred)

    req_vol_mase = mase(y_test_req_vol, req_vol_pred)
    resp_time_mase = mase(y_test_resp_time, resp_time_pred)
    err_cnt_mase = mase(y_test_err_cnt, err_cnt_pred)

    return req_vol_rmse,resp_time_rmse,err_cnt_rmse,req_vol_mae,resp_time_mae,err_cnt_mae,req_vol_mape,resp_time_mape,err_cnt_mape,req_vol_mase,resp_time_mase,err_cnt_mase


def predict_test(X_test,FEATURES,reg_req_vol,reg_resp_time,reg_err_cnt):
    # X_test = pd.DataFrame(X_test, columns=FEATURES)  # Assuming FEATURES is a list of feature names

    req_vol_pred = reg_req_vol.predict(X_test)
    resp_time_pred = reg_resp_time.predict(X_test)
    err_cnt_pred = reg_err_cnt.predict(X_test)

    # # Calculate prediction intervals
    # req_vol_interval = calculate_prediction_interval(X_train,y_train_req_vol,X_test)
    # resp_time_interval = calculate_prediction_interval(X_train,y_train_resp_time,X_test)
    # err_cnt_interval = calculate_prediction_interval(X_train,y_train_err_cnt,X_test)

    # Calculate prediction intervals
    req_vol_interval = calculate_prediction_interval(req_vol_pred)
    resp_time_interval = calculate_prediction_interval(resp_time_pred)
    err_cnt_interval = calculate_prediction_interval(err_cnt_pred)
    return req_vol_pred,resp_time_pred,err_cnt_pred,req_vol_interval,resp_time_interval,err_cnt_interval

def classify_services(data,correlation_threshold,inp):
      # Set the correlation threshold to classify as LR
    if inp=='vol_err':
        # if 'error_count' in data.columns:
            # Calculate the correlation between total count and error count for each service
        correlations = data.groupby('sid')[
                               ['total_req_count', 'error_count']].corr().iloc[1::2, 0]
    if inp=='vol_resp':
        # if 'resp_time' in data.columns:
            # Calculate the correlation between total count and resp_time for each service
        correlations = data.groupby('sid')[
                               ['total_req_count', 'resp_time_sum']].corr().iloc[1::2, 0]

    # Determine LR and NLR services based on highest correlation
    highest_correlations = correlations.groupby('sid').max()

    lr_services = set()
    nlr_services = set()

    # Dictionary to store maximum correlation for each service
    max_correlations = {}

    # Include services with similar highest correlation to LR or NLR
    for service, max_correlation in highest_correlations.items():
        correlated_services = correlations[correlations.index.get_level_values('sid') == service]

        if not pd.isnull(max_correlation) and round(max_correlation, 1) >= correlation_threshold:
            lr_services.add(service)
            lr_services.update(correlated_services[correlated_services >= correlation_threshold].index)
            # Store the maximum correlation for this service
            max_correlations[service] = max_correlation
        else:
            nlr_services.add(service)
            nlr_services.update(correlated_services[correlated_services < correlation_threshold].index)
            max_correlations[service] = max_correlation
    # Create DataFrames for LR and NLR services
    lr_data = data[data['sid'].isin(lr_services)].copy()
    nlr_data = data[data['sid'].isin(nlr_services)].copy()

    # # Add a new column for maximum correlation
    # lr_data['correlation'] = lr_data['calcFields.service_name'].map(max_correlations)
    # nlr_data['correlation'] = nlr_data['calcFields.service_name'].map(max_correlations)

    return lr_data, nlr_data, max_correlations

def list_unique_services(data, max_correlations):
    data = data.dropna(subset=['sid']).copy()

    # Get unique service names
    unique_services = data['sid'].unique()
    # Create a DataFrame with the appropriate column name
    unique_services_df = pd.DataFrame({'sid': unique_services})
    # Add maximum correlation for each service
    unique_services_df['Maximum Correlation'] = unique_services_df['sid'].map(max_correlations)

    # Sort the DataFrame based on 'Maximum Correlation' in descending order
    unique_services_df = unique_services_df.sort_values(by='Maximum Correlation', ascending=False)

    # Print count of unique services
    print(f'Count of services: {len(unique_services)}')

    return unique_services_df






def train_and_predict(filtered_df,sid):
    # train_and_forecast_service(filtered_df, sid)
    model_req_vol, model_resp_time, model_err_cnt = train_service_model(filtered_df, sid)
    # service_name = service_name.replace('/', '_')
    model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
    model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
    model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))
    print(f"Models saved for {sid} ")
    if model_req_vol is not None:
        print("inside forecast phase")
        pred_start_date='2023-12-01 11:00:00'
        pred_end_date='2023-12-20 11:00:00'
        freq='5T'
        forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
        print("next going into bulk save")
        bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
                                 elk_password)
def save_csv(csv_name,df):
    csv_file_path =os.path.join(project_path, training_path, csv_name)  # Provide the desired file path
    df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to: {csv_file_path}")
    logging.info(f"DataFrame saved to: {csv_file_path}")

from concurrent.futures import ThreadPoolExecutor

# @appp.route('/train')
# def train():
#     st_time = timer()
#     # To ignore specific warnings
#     warnings.filterwarnings("ignore", category=FutureWarning)
#
#     color_pal = sns.color_palette()
#     plt.style.use('fivethirtyeight')
#
#     start_time = 1701282600000
#     end_time = 1701369000000
#     # df = connectelk_retrive_data(app_id, start_time, end_time, source_index)
#     csv_file_path = os.path.join(project_path, training_path, f'{source_index}.csv')
#
#     df = pd.read_csv(csv_file_path)
#
#     print(df.head())
#
#     logging.info(df.head())
#
#     correlation_threshold = 0.7
#
#     vec_data, venc_data, max_correlations_vol_err = classify_services(df, correlation_threshold, 'vol_err')
#     vrc_data, vrnc_data, max_correlations_vol_resp = classify_services(df, correlation_threshold, 'vol_resp')
#
#     df_corr_vol_err = list_unique_services(vec_data, max_correlations_vol_err)
#     df_corr_vol_resp = list_unique_services(vec_data, max_correlations_vol_resp)
#     save_csv('vol_err_corr_services.csv', df_corr_vol_err)  # Provide the desired file path
#     save_csv('vol_resp_corr_services.csv', df_corr_vol_resp)  # Provide the desired file path
#
#     df_all_services = update_unique_services(df)
#     # Sort the DataFrame by 'sid' in ascending order
#     df_all_services = df_all_services.sort_values(by='sid')
#     save_csv('all_services.csv', df_all_services)
#
#     # services_all = df_all_services['sid'].tolist()[:5]
#     services_all = [1034,1036,1039,1051,1103,1059,1100]
#
#     services_err = df_corr_vol_err['sid'].tolist()[:5]
#     services_resp = df_corr_vol_resp['sid'].tolist()[:5]
#
#
#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         # Train and predict for volume
#         executor.map(train_and_predict_volume, [(df[df['sid'] == sid], sid) for sid in services_all])
#
#         # Train and predict for error count (only for correlated services)
#         executor.map(train_and_predict_error_count, [(df[df['sid'] == sid], sid) for sid in services_err])
#
#         # Train and predict for response time (only for correlated services)
#         executor.map(train_and_predict_response_time, [(df[df['sid'] == sid], sid) for sid in services_resp])
#
#     ed_time = timer()
#     total_execution_time = ed_time - st_time
#
#     # Example response
#     response_data = {'status': 'success', 'message': 'Model trained successfully',
#                      'total_execution_time': total_execution_time}
#     logging.info(response_data)
#
#     # Return a valid HTTP response (JSON in this case)
#     return jsonify(response_data)
#
#
# def train_and_predict_volume(args):
#     filtered_df, sid = args
#     train_and_predict_service(filtered_df, sid, 'req_vol')
#
#
# def train_and_predict_error_count(args):
#     filtered_df, sid = args
#     train_and_predict_service(filtered_df, sid, 'err_cnt')
#
#
# def train_and_predict_response_time(args):
#     filtered_df, sid = args
#     train_and_predict_service(filtered_df, sid, 'resp_time')
#

# def train_and_predict_service(filtered_df, sid, target):
#     try:
#         model_req_vol, model_resp_time, model_err_cnt = train_and_forecast_service(filtered_df, sid)
#
#         if model_req_vol is not None:
#             # Save the volume model
#             model_req_vol.save_model(os.path.join(project_path, models_path, f'{sid}_req_vol.json'))
#             logging.info(f"Volume model for {sid} saved successfully.")
#
#             # Predict future volume data
#             if target == 'req_vol':
#                 pred_start_date = '2023-12-11 11:00:00'
#                 pred_end_date = '2023-12-12 11:00:00'
#                 freq = '5T'
#                 forecast = forecast_future(app_id, pred_start_date, pred_end_date, freq, filtered_df, sid)
#
#                 # Save the volume forecast data
#                 bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
#                                          elk_password)
#                 logging.info(f"Volume forecast for {sid} saved successfully.")
#
#         if model_resp_time is not None and target == 'resp_time':
#             # Save the response time model
#             model_resp_time.save_model(os.path.join(project_path, models_path, f'{sid}_resp_time.json'))
#             logging.info(f"Response time model for {sid} saved successfully.")
#
#             # Predict future response time data
#             pred_start_date = '2023-12-11 11:00:00'
#             pred_end_date = '2023-12-12 11:00:00'
#             freq = '5T'
#             forecast = forecast_future(app_id, pred_start_date, pred_end_date, freq, filtered_df, sid)
#
#             # Save the response time forecast data
#             bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
#                                      elk_password)
#             logging.info(f"Response time forecast for {sid} saved successfully.")
#
#         if model_err_cnt is not None and target == 'err_cnt':
#             # Save the error count model
#             model_err_cnt.save_model(os.path.join(project_path, models_path, f'{sid}_err_cnt.json'))
#             logging.info(f"Error count model for {sid} saved successfully.")
#
#             # Predict future error count data
#             pred_start_date = '2023-12-11 11:00:00'
#             pred_end_date = '2023-12-12 11:00:00'
#             freq = '5T'
#             forecast = forecast_future(app_id, pred_start_date, pred_end_date, freq, filtered_df, sid)
#
#             # Save the error count forecast data
#             bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
#                                      elk_password)
#             logging.info(f"Error count forecast for {sid} saved successfully.")
#
#     except Exception as e:
#         logging.error(f"Error processing service {sid}: {e}")

# def forecast_future(app_id, start_date, end_date, freq, filtered_df, sid=None):
#     model_req_vol, model_resp_time, model_err_cnt = load_input_model(sid)
#
#     if model_req_vol:
#         actual_future_dataset = predict_future_dataset(app_id, model_req_vol, model_resp_time, model_err_cnt, sid, freq,
#                                                        start_date, end_date, filtered_df)
#         return actual_future_dataset
#     else:
#         print("Error: Unable to proceed with prediction. Check the availability of volume model.")
#         return pd.DataFrame()

# Rest of the code remains unchanged


@ap.route('/train')
def train():
    st_time=timer()
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')

    # start_time=1701282600000
    # end_time= 1701369000000
    # start_time=1701408600000
    # end_time= 1701495000000
    # df=connectelk_retrive_data(app_id,start_time,end_time,source_index)
    # save_csv(f'{source_index}.csv',df)
    # csv_file_path = os.path.join(project_path, training_path,'output_selected_dataframe.csv')
    csv_file_path = os.path.join(project_path, predictions_path,'1007_predicted.csv')
    #
    df = pd.read_csv(csv_file_path)


    bulk_save_forecast_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username,elk_password)
    print(df.head())

    logging.info(df.head())

    correlation_threshold = 0.7

    vec_data, venc_data, max_correlations_vol_err = classify_services(df,correlation_threshold,'vol_err')
    vrc_data, vrnc_data, max_correlations_vol_resp = classify_services(df,correlation_threshold,'vol_resp')


    df_corr_vol_err = list_unique_services(vec_data, max_correlations_vol_err)
    df_corr_vol_resp= list_unique_services(vec_data, max_correlations_vol_resp)
    save_csv( 'vol_err_corr_services.csv',df_corr_vol_err)# Provide the desired file path
    save_csv( 'vol_resp_corr_services.csv',df_corr_vol_resp)# Provide the desired file path


    df_all_services=update_unique_services(df)
   # Sort the DataFrame by 'sid' in ascending order
    df_all_services= df_all_services.sort_values(by='sid')
    save_csv('all_services.csv',df_all_services)

    services = df_all_services['sid'].tolist()[:1]

    def train_predict_for_service(sid):

        # Train the model
        filtered_df = df[df['sid'] == sid]
        print(filtered_df.head())
        filtered_df= filtered_df.sort_values(by='record_time')
        save_csv(f'{sid}_train.csv',filtered_df)
        train_and_predict(filtered_df,sid)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(train_predict_for_service, services)

    ed_time=timer()
    total_execution_time = ed_time - st_time

    # Example response
   #  response_data = {'status': 'success', 'message': 'Model trained successfully', 'total_execution_time': total_execution_time}
    response_data = {'status': 'success', 'message': 'prediction data saved successfully'}
   #
   #  logging.info(response_data)
   #
   #  # Return a valid HTTP response (JSON in this case)
    return jsonify(response_data)
   #
   #
   #

@ap.route('/retrain')
def retrain():
    st_time=timer()
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')
    # app_id="753"

    # date=datetime.now() - timedelta(days=1)
    # start_time = date.timestamp() * 1000  # Convert to milliseconds
    # end_time = (date + timedelta(days=1)).timestamp() * 1000
    start_time= 1700179200000
    end_time=1701801000000
    print(start_time)
    print(end_time)
    df = connectelk_retrive_data(app_id, start_time, end_time,source_index)

    if df is not None:
        print(df.head())

        daily_unq_services=update_unique_services(df)
        daily_unq_services= daily_unq_services.sort_values(by='sid')
        csv_file_path = os.path.join(project_path,training_path,'daily_services.csv') # Provide the desired file path
        daily_unq_services.to_csv(csv_file_path, index=False)
        print(f"DataFrame saved to: {csv_file_path}")
        logging.info(f"DataFrame saved to: {csv_file_path}")

        services = daily_unq_services['sid'].tolist()[:5]
        def retrain_model_for_service(sid):


            # RETrain the model
            filtered_df = df[df['sid'] == sid]
            print(filtered_df.head())
            filtered_df= filtered_df.sort_values(by='record_time')

            csv_file_path = os.path.join(project_path, retraining_path,f'{sid}_retrain.csv')  # Provide the desired file path
            filtered_df.to_csv(csv_file_path, index=False)
            print(f"DataFrame saved to: {csv_file_path}")
            logging.info(f"DataFrame saved to: {csv_file_path}")

            if not filtered_df.empty:
                print(filtered_df.head())
                # check the model exists
                model_req_vol = os.path.join(project_path, models_path,f'{sid}_req_vol.json')  # Update with the actual file names
                model_resp_time = os.path.join(project_path, models_path,f'{sid}_resp_time.json')
                model_err_cnt = os.path.join(project_path, models_path,f'{sid}_err_cnt.json')
                if os.path.exists(model_req_vol) and os.path.exists(model_err_cnt) and os.path.exists(model_resp_time):
                    model_req_vol, model_resp_time, model_err_cnt= grid_search_retrain_model(model_req_vol,
                                                                                             model_err_cnt,
                                                                                             model_resp_time,
                                                                                             filtered_df, sid)
                    # Save the updated model
                    model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
                    model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
                    model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))

                    pred_start_date='2023-12-11 11:00:00'
                    pred_end_date='2023-12-12 11:00:00'
                    freq='5T'
                    forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
                    # save_forecast_index(forecast,target_index)
                    bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
                                             elk_password)


                else:
                    print(f"Model files not found for {sid}. Train new model first.")
                    train_and_predict(filtered_df, sid)
                    # model_req_vol, model_resp_time, model_err_cnt = train_and_forecast_service(filtered_df, sid)
                    # model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
                    # model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
                    # model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))
                    # if model_req_vol is not None:
                    #
                    #     pred_start_date='2023-12-11 11:00:00'
                    #     pred_end_date='2023-12-22 11:00:00'
                    #     freq='5T'
                    #     forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
                    #     # save_forecast_index(forecast,target_index)
                    #     bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

            # all_services_df = pd.read_csv(os.path.join(project_path, training_path, 'all_services.csv'))
            # all_services = all_services_df['sid']
            #
            # for sid in all_services[:3]:
            #     if sid not in daily_services[:3]:  # Avoid predicting for services already processed during retraining
            #         # Load or check the existence of the model
            #         model_req_vol = os.path.join(project_path, models_path, f'{sid}_req_vol.json')
            #         model_err_cnt = os.path.join(project_path, models_path, f'{sid}_err_cnt.json')
            #         model_resp_time = os.path.join(project_path, models_path, f'{sid}_resp_time.json')
            #
            #         if os.path.exists(model_req_vol) and os.path.exists(model_err_cnt) and os.path.exists(model_resp_time):
            #             # Predict the future data set
            #             pred_start_date = '2023-11-22 11:00:00'
            #             pred_end_date = '2023-12-11 11:00:00'
            #             freq = '5T'
            #             # filtered_df = connectelk_retrive_data(app_id, start_time, end_time)  # Adjust start_time and end_time as needed
            #             forecast_future(app_id, pred_start_date, pred_end_date, freq,sid)
            #         else:
            #             print(f"Model files not found for {sid}. Train new model first.")
            else:
                print("No records found for retraining model")
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(retrain_model_for_service, services)#
        ed_time=timer()
        total_execution_time = ed_time - st_time

        # Example response
        response_data = {'status': 'success', 'message': 'Model retrained successfully', 'total_execution_time': total_execution_time}
        logging.info(response_data)

    # Return a valid HTTP response (JSON in this case)
        return jsonify(response_data)
    else:
        logging.error("Failed to load data.")
        return None
def train_service_model(df,sid=None):

    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df['record_time'] = df['record_time'].astype(np.int64) // 10**6  # Convert nanoseconds to milliseconds
    start_time = timer()

    df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')

    df.set_index('record_time', inplace=True)


    df=remove_outliers(df)
    train,test,df=train_test_splitdata(df)

    df = create_features(df)

    df = add_lags(df)
    if len(df) <4:
        print(f"Not enough samples for {sid} time series cross-validation. Please add more data.")
    else:
        # reg_req_vol,reg_resp_time,reg_err_cnt,rmses,maes,mapes=train_model(df)
        reg_req_vol,reg_resp_time,reg_err_cnt,rmses,maes,mapes,mases=train_model_with_grid_search(df)

        if reg_req_vol is not None:

            print_metrics(sid,rmses,'RMSE')
            print_metrics(sid,maes,'MAE')
            print_metrics(sid,mapes,'MAPE')
            print_metrics(sid,mases,'MASE')


            # Print a message indicating the end of the training and forecasting process

            print(df.index.max())

            end_time = timer()
            training_time = end_time - start_time
            print(f"Training completed for {sid}. Training time: {training_time} seconds.")
            return reg_req_vol,reg_resp_time,reg_err_cnt
            # return reg_req_vol,reg_resp_time,reg_err_cnt
    # Schedule the future prediction using the saved models
    # schedule.every().day.at("14:43").do(predict_future_dataset, model_req_vol, model_err_cnt, model_resp_time, service_name)

    # predict_future_dataset(model_req_vol,model_err_cnt,model_resp_time,service_name)
    # return model_req_vol,model_err_cnt,model_resp_time
    return None,None,None
def grid_search_retrain_model(model_req_vol, model_err_cnt, model_resp_time, df,sid):


        df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')

        df.set_index('record_time', inplace=True)

        # Set the index to the 'datetime' column
        #     df.set_index('datetime', inplace=True)
        df=remove_outliers(df)
        train,test,df=train_test_splitdata(df)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes=[]
        mapes=[]
        mases=[]

        # Define the hyperparameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [500,800,1000],
            'learning_rate': [0.01, 0.2, 0.3],
            'max_depth': [13,15,20],
            # Add other hyperparameters to tune
        }
        for train_idx, val_idx in tss.split(df):

            req_vol_preds = []
            resp_time_preds = []
            err_cnt_preds=[]

            start_time = timer()

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
            # 'lag1','lag2','lag3']
            REQ_VOL_TARGET = 'total_req_count'
            RESP_TIME_TARGET = 'resp_time_sum'
            ERR_CNT_TARGET = 'error_count'


            # Add lags and create features
            df = add_lags(df)  # Assuming you have a function to add lags
            df = create_features(df)
            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']

            new_X_req_vol = test[FEATURES]
            new_y_req_vol = test[REQ_VOL_TARGET]
            dnew_req_vol = xgb.DMatrix(new_X_req_vol, label=new_y_req_vol)
            updated_model_req_vol = xgb.XGBRegressor(objective='reg:squarederror')

            grid_search_req_vol = GridSearchCV(updated_model_req_vol, param_grid, scoring='neg_mean_squared_error', cv=2)
            grid_search_req_vol.fit(new_X_req_vol, new_y_req_vol)
            updated_model_req_vol = grid_search_req_vol.best_estimator_


            new_X_resp_time = test[FEATURES]
            new_y_resp_time = test[RESP_TIME_TARGET]
            dnew_resp_time = xgb.DMatrix(new_X_resp_time, label=new_y_resp_time)
            updated_model_resp_time = xgb.XGBRegressor(objective='reg:squarederror')

            grid_search_resp_time = GridSearchCV(updated_model_resp_time, param_grid, scoring='neg_mean_squared_error', cv=2)
            grid_search_resp_time.fit(new_X_resp_time, new_y_resp_time)
            updated_model_resp_time = grid_search_resp_time.best_estimator_


            new_X_err_cnt = test[FEATURES]
            new_y_err_cnt = test[ERR_CNT_TARGET]
            dnew_err_cnt = xgb.DMatrix(new_X_err_cnt, label=new_y_err_cnt)
            updated_model_err_cnt = xgb.XGBRegressor(objective='reg:squarederror')

            grid_search_err_cnt = GridSearchCV(updated_model_err_cnt, param_grid, scoring='neg_mean_squared_error', cv=2)
            grid_search_err_cnt.fit(new_X_err_cnt, new_y_err_cnt)
            updated_model_err_cnt = grid_search_err_cnt.best_estimator_


            # Print updated model parameters
            print("Updated Model Parameters:")
            # print(updated_model_req_vol.get_booster().get_dump()[0])
            # Get the best hyperparameters
            best_params_req_vol = grid_search_req_vol.best_params_
            best_params_resp_time = grid_search_resp_time.best_params_
            best_params_err_cnt = grid_search_err_cnt.best_params_
            print("best parameters for volume, response time and error count",best_params_req_vol,best_params_resp_time,best_params_err_cnt)

            # Make predictions with the updated model
            pred_req_vol = updated_model_req_vol.predict(new_X_req_vol)
            pred_resp_time = updated_model_resp_time.predict(new_X_resp_time)
            pred_err_cnt = updated_model_err_cnt.predict(new_X_err_cnt)



            # test = test[:len(pred_req_vol)]
            req_vol_rmse,resp_time_rmse,err_cnt_rmse,req_vol_mae,resp_time_mae,err_cnt_mae,req_vol_mape,resp_time_mape,err_cnt_mape,req_vol_mase,resp_time_mase,err_cnt_mase=evaluate_metrics(new_y_req_vol, pred_req_vol,new_y_resp_time, pred_resp_time,new_y_err_cnt, pred_err_cnt)

            rmses.append((req_vol_rmse, resp_time_rmse,err_cnt_rmse))
            maes.append((req_vol_mae,resp_time_mae,err_cnt_mae))
            mapes.append((req_vol_mape,resp_time_mape,err_cnt_mape))
            mases.append((req_vol_mase,resp_time_mase,err_cnt_mase))

        print_metrics(sid,rmses,'RMSE')
        print_metrics(sid,maes,'MAE')
        print_metrics(sid,mapes,'MAPE')
        print_metrics(sid,mases,'MASE')

        end_time = timer()
        training_time = end_time - start_time
        print(f"Reraining completed for {sid}. Training time: {training_time} seconds.")


        return updated_model_req_vol,updated_model_resp_time,updated_model_err_cnt



def forecast_future(app_id,start_date,end_date,freq,filtered_df,sid=None):
    model_req_vol,model_resp_time,model_err_cnt= load_input_model(sid)
    print(f"Models loaded for {sid}")
    if model_req_vol and model_resp_time and model_err_cnt:
        print("going to predict")
        future_dataset = predict_future_dataset(app_id, model_req_vol, model_resp_time, model_err_cnt, sid, freq, start_date, end_date, filtered_df)
        print("saving to csv")
        csv_file_path = os.path.join(project_path, predictions_path,f'{sid}_predicted.csv')  # Provide the desired file path
        future_dataset.to_csv(csv_file_path, index=False)
        print(f"Prediction DataFrame saved to: {csv_file_path}")
    else:
        print("Error: Unable to proceed with prediction. Check the availability of model files.")
    return future_dataset


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train or retrain the model.')
    # parser.add_argument('mode', choices=['train', 'retrain'], default='train', help='Specify train or retrain mode')
    #
    # args = parser.parse_args()
    # main(args.mode)
    ap.run(debug=True)



