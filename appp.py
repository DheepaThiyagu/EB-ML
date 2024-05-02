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
import logging
from xgboost import XGBRegressor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
appp = Flask(__name__)
from concurrent.futures import ThreadPoolExecutor
@appp.route('/')
def index():
    return 'Welcome to your EB_ML Flask App!'

project_path = os.getcwd()

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
# Specify the relative path to your models directory



elasticsearch_host = config.get('elasticsearch', 'host')
elasticsearch_port = config.getint('elasticsearch', 'port')
elk_username = config.get('elasticsearch', 'username')
elk_password = config.get('elasticsearch', 'password')

target_index = config.get('indices', 'target_index')
training_index = config.get('indices', 'training_index')

models_path = config.get('paths', 'models_path')
predictions_path = config.get('paths', 'predictions_path')
training_path = config.get('paths', 'training_path')
retraining_path = config.get('paths', 'retraining_path')

app_id="753"
source_index='fivemins_wm_oseb1decv1_753'
# target_index='predictions_9_jan_by_script'


def train_volume_model_with_grid_search(df,sid):
    print(f"Training started for {sid} for  Request Volume")
    start_time = timer()
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes = []
        mapes = []
        # mases = []

        for train_idx, val_idx in tss.split(df):
            req_vol_preds = []

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
            REQ_VOL_TARGET = 'total_req_count'

            X_train = train[FEATURES]
            y_train_req_vol = train[REQ_VOL_TARGET]

            X_test = test[FEATURES]
            y_test_req_vol = test[REQ_VOL_TARGET]

            # # Define the hyperparameter grid for GridSearchCV
            # param_grid = {
            #     'n_estimators': [500, 800, 1000],
            #     'learning_rate': [0.01, 0.2, 0.3],
            #     'max_depth': [13, 15, 20],
            #     # Add other hyperparameters to tune
            # }
            #
            # # Create XGBoost regressor model
            # reg_req_vol = xgb.XGBRegressor(base_score=0.5, booster='gbtree', objective='reg:squarederror')
            #
            # # Create a scorer for GridSearchCV using negative mean squared error
            # scorer = make_scorer(mean_squared_error, greater_is_better=False)
            #
            # # Create GridSearchCV instance for volume regressor
            # grid_search_req_vol = GridSearchCV(reg_req_vol, param_grid, scoring=scorer, cv=tss)
            #
            # # Fit GridSearchCV to the training data
            # grid_search_req_vol.fit(X_train, y_train_req_vol.astype(int))
            #
            # # Get the best hyperparameters
            # best_params_req_vol = grid_search_req_vol.best_params_
            # print("best parameters for volume:", best_params_req_vol)
            #
            # # Train the final model with the best hyperparameters
            # reg_req_vol = xgb.XGBRegressor(objective='reg:squarederror', **best_params_req_vol)
            # Define the hyperparameters
            params = {
                'objective': 'reg:squarederror',
                'base_score': 0.5,
                'booster': 'gbtree',
                'n_estimators': 1000,
                'learning_rate': 0.2,
                'max_depth': 15,
                # Add other hyperparameters as needed
            }

            # Create XGBoost regressor model with predefined hyperparameters
            reg_req_vol = xgb.XGBRegressor(**params)

            fit_req_vol_model(X_train, y_train_req_vol, X_test, y_test_req_vol, reg_req_vol)

            req_vol_pred, req_vol_interval = predict_req_vol( reg_req_vol,X_test)

            # Append intervals to your lists
            req_vol_preds.append((req_vol_pred, req_vol_interval))

            # # Create a DataFrame to store predictions and intervals
            # predictions_df = pd.DataFrame({
            #     'Datetime': X_test.index,
            #     'Req_Vol_Pred': req_vol_pred,
            #     'Req_Vol_Lower': req_vol_interval[0],
            #     'Req_Vol_Upper': req_vol_interval[1],
            #     'Req_Vol_Confidence_Score': req_vol_interval[2],
            # })
            #
            # # Save the DataFrame to a CSV file
            # predictions_df.to_csv('volume_predictions.csv', index=False)

            req_vol_rmse, req_vol_mae, req_vol_mape=evaluate_metrics(y_test_req_vol, req_vol_pred)

            rmses.append((req_vol_rmse,))
            maes.append((req_vol_mae,))
            mapes.append((req_vol_mape,))
            # mases.append((req_vol_mase,))

        print_metrics(sid,rmses,'RMSE','Request Volume')
        print_metrics(sid,maes,'MAE','Request Volume')
        print_metrics(sid,mapes,'MAPE','Request Volume')
        # print_metrics(sid,mases,'MASE','Request Volume')

        end_time = timer()
        training_time = end_time - start_time
        print(f"Training completed for {sid} for Request Volume. Training time: {training_time} seconds.\n\n\n")

        return reg_req_vol


    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service{sid} due to error: {e}")

        # Return the final model
    return None


def train_response_time_model_with_grid_search(df,sid):
    print(f"Training started for {sid} for Response Time")

    start_time = timer()

    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes = []
        mapes = []
        # mases = []

        for train_idx, val_idx in tss.split(df):
            resp_time_preds = []

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
            RESP_TIME_TARGET = 'resp_time_sum'

            X_train = train[FEATURES]
            y_train_resp_time = train[RESP_TIME_TARGET]

            X_test = test[FEATURES]
            y_test_resp_time = test[RESP_TIME_TARGET]

            # # Define the hyperparameter grid for GridSearchCV
            # param_grid = {
            #     'n_estimators': [500, 800, 1000],
            #     'learning_rate': [0.01, 0.2, 0.3],
            #     'max_depth': [13, 15, 20],
            #     # Add other hyperparameters to tune
            # }
            #
            # # Create XGBoost regressor model
            # reg_resp_time = xgb.XGBRegressor(base_score=0.5, booster='gbtree', objective='reg:squarederror')
            #
            # # Create a scorer for GridSearchCV using negative mean squared error
            # scorer = make_scorer(mean_squared_error, greater_is_better=False)
            #
            # # Create GridSearchCV instance for response time regressor
            # grid_search_resp_time = GridSearchCV(reg_resp_time, param_grid, scoring=scorer, cv=tss)
            #
            # # Fit GridSearchCV to the training data
            # grid_search_resp_time.fit(X_train, y_train_resp_time)
            #
            # # Get the best hyperparameters
            # best_params_resp_time = grid_search_resp_time.best_params_
            # print("best parameters for response time:", best_params_resp_time)
            #
            # # Train the final model with the best hyperparameters
            # reg_resp_time = xgb.XGBRegressor(objective='reg:squarederror', **best_params_resp_time)
            # Define the hyperparameters
            params = {
                'objective': 'reg:squarederror',
                'base_score': 0.5,
                'booster': 'gbtree',
                'n_estimators': 1000,
                'learning_rate': 0.2,
                'max_depth': 15,
                # Add other hyperparameters as needed
            }

            # Create XGBoost regressor model with predefined hyperparameters
            reg_resp_time = xgb.XGBRegressor(**params)
            fit_resp_time_model(X_train, y_train_resp_time, X_test, y_test_resp_time, reg_resp_time)
            resp_time_pred,resp_time_interval=predict_resp_time(reg_resp_time, X_test)
            # Append intervals to your lists
            resp_time_preds.append((resp_time_pred, resp_time_interval))

            # Create a DataFrame to store predictions and intervals
            # predictions_df = pd.DataFrame({
            #     'Datetime': X_test.index,
            #     'Resp_Time_Pred': resp_time_pred,
            #     'Resp_Time_Lower': resp_time_interval[0],
            #     'Resp_Time_Upper': resp_time_interval[1],
            #     'Resp_Time_Confidence_Score': resp_time_interval[2],
            # })

            # Save the DataFrame to a CSV file
            # predictions_df.to_csv('response_time_predictions.csv', index=False)

            resp_time_rmse, resp_time_mae, resp_time_mape=evaluate_metrics(y_test_resp_time, resp_time_pred)
            rmses.append((resp_time_rmse,))
            maes.append((resp_time_mae,))
            mapes.append((resp_time_mape,))
            # mases.append((resp_time_mase,))

        print_metrics(sid,rmses,'RMSE','Response Time')
        print_metrics(sid,maes,'MAE','Response Time')
        print_metrics(sid,mapes,'MAPE','Response Time')
        # print_metrics(sid,mases,'MASE','Response Time')

        # Print a message indicating the end of the training process

        print(df.index.max())

        end_time = timer()
        training_time = end_time - start_time
        print(f"Training completed for {sid} for Response time. Training time: {training_time} seconds.")
        # Return the final model
        return reg_resp_time

    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service{sid} due to error: {e}")

    return None


def train_error_count_model_with_grid_search(df,sid):
    print(f"Training started for {sid} for  Error Count")

    start_time = timer()
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes = []
        mapes = []
        # mases = []

        for train_idx, val_idx in tss.split(df):
            err_cnt_preds = []

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
            ERR_CNT_TARGET = 'error_count'

            X_train = train[FEATURES]
            y_train_err_cnt = train[ERR_CNT_TARGET]

            X_test = test[FEATURES]
            y_test_err_cnt = test[ERR_CNT_TARGET]

            # # Define the hyperparameter grid for GridSearchCV
            # param_grid = {
            #     'n_estimators': [500, 800, 1000],
            #     'learning_rate': [0.01, 0.2, 0.3],
            #     'max_depth': [13, 15, 20],
            #     # Add other hyperparameters to tune
            # }
            #
            # # Create XGBoost regressor model
            # reg_err_cnt = xgb.XGBRegressor(base_score=0.5, booster='gbtree', objective='reg:squarederror')
            #
            # # Create a scorer for GridSearchCV using negative mean squared error
            # scorer = make_scorer(mean_squared_error, greater_is_better=False)
            #
            # # Create GridSearchCV instance for error count regressor
            # grid_search_err_cnt = GridSearchCV(reg_err_cnt, param_grid, scoring=scorer, cv=tss)
            #
            # # Fit GridSearchCV to the training data
            # grid_search_err_cnt.fit(X_train, y_train_err_cnt.astype(int))
            #
            # # Get the best hyperparameters
            # best_params_err_cnt = grid_search_err_cnt.best_params_
            # print("best parameters for error count:", best_params_err_cnt)
            #
            # # Train the final model with the best hyperparameters
            # reg_err_cnt = xgb.XGBRegressor(objective='reg:squarederror', **best_params_err_cnt)
            # Define the hyperparameters
            params = {
                'objective': 'reg:squarederror',
                'base_score': 0.5,
                'booster': 'gbtree',
                'n_estimators': 1000,
                'learning_rate': 0.2,
                'max_depth': 15,
                # Add other hyperparameters as needed
            }

            # Create XGBoost regressor model with predefined hyperparameters
            reg_err_cnt = xgb.XGBRegressor(**params)
            fit_err_cnt_model(X_train,y_train_err_cnt, X_test, y_test_err_cnt, reg_err_cnt)
            err_cnt_pred,err_cnt_interval = predict_err_cnt(reg_err_cnt, X_test)
            # Append intervals to your lists
            err_cnt_preds.append((err_cnt_pred, err_cnt_interval))

            # Create a DataFrame to store predictions and intervals
            # predictions_df = pd.DataFrame({
            #     'Datetime': X_test.index,
            #     'Err_Cnt_Pred': err_cnt_pred,
            #     'Err_Cnt_Lower': err_cnt_interval[0],
            #     'Err_Cnt_Upper': err_cnt_interval[1],
            #     'Err_Cnt_Confidence_Score': err_cnt_interval[2],
            # })

            # Save the DataFrame to a CSV file
            # predictions_df.to_csv('error_count_predictions.csv', index=False)

            err_cnt_rmse, err_cnt_mae,err_cnt_mape = evaluate_metrics(y_test_err_cnt, err_cnt_pred)

            rmses.append((err_cnt_rmse,))
            maes.append((err_cnt_mae,))
            mapes.append((err_cnt_mape,))
            # mases.append((err_cnt_mase,))
        print_metrics(sid,rmses,'MAE','Error Count')
        print_metrics(sid,maes,'MAE','Error Count')
        print_metrics(sid,mapes,'MAPE','Error Count')
        # print_metrics(sid,mases,'MASE','Error Count')

        # Print a message indicating the end of the training process

        print(df.index.max())

        end_time = timer()
        training_time = end_time - start_time
        print(f"Training completed for {sid} for Error count. Training time: {training_time} seconds.")
        return reg_err_cnt

    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service {sid} due to error: {e}")

        # Return the final model
    return None



def predict_future_dataset(app_id,sid,freq,start_date,end_date,df_corr_vol_err,df_corr_vol_resp):
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
    pred_err = xgb.XGBRegressor()
    pred_resp = xgb.XGBRegressor()

    model_req_vol = os.path.join(project_path, models_path, f'{sid}_vol.json')
    model_resp_time = os.path.join(project_path, models_path, f'{sid}_resp.json')
    model_err_cnt = os.path.join(project_path, models_path, f'{sid}_err.json')

    try:
        if os.path.exists(model_req_vol):
            pred_vol.load_model(model_req_vol)
        else:
            model_req_vol=None
            print(f" Request volume model not found for {sid}")



        if os.path.exists(model_err_cnt):
            if sid in df_corr_vol_err:
                pred_err.load_model(model_err_cnt)
            else:
                model_err_cnt=None
                print(f"Model available but {sid} not in vol_err correlation list now. So not retraining and predicting now")
        else:
            model_err_cnt=None
            print(f" Error count model not found for {sid} as this service is not listed in vol_err corr list ever")



        if os.path.exists(model_resp_time):
            if sid in df_corr_vol_resp:
                 pred_resp.load_model(model_resp_time)
            else:
                model_resp_time=None
                print(f"Model available but {sid} not in vol_resp correlation list now. So not retraining and predicting now")
        else:
            model_resp_time=None
            print(f" Response time model not found for {sid} as this service is not listed in vol_resp corr list ever")

    except Exception as e:
        print(f"Error loading models: {e}")



    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
    # 'lag1','lag2','lag3']
    # REQ_VOL_TARGET = 'req_vol'
    # RESP_TIME_TARGET = 'resp_time'
    # ERR_CNT_TARGET = 'err_cnt'



    if model_req_vol is not None:
        # Create future dataframe
        print( f"creating  prediction data for {sid} for volume")
        future = pd.date_range(start_date,end_date, freq=freq) # freq is set for data granularity ( 5 mins or 1 hour)
        future_df = pd.DataFrame(index=future)

        future_w_features=create_features(future_df)
        # Convert app_id to a pandas Series
        app_id_series = pd.Series(app_id)

        # future_w_features['datetime']=pd.to_datetime(future_w_features['datetime'],format="%Y-%m-%d %H:%M:%S")
        future_w_features['datetime']=future_w_features.index

        future_w_features['app_id']=app_id_series.iloc[0]

        future_w_features['sid']=sid
        future_w_features['pred_req_vol'] = pred_vol.predict(future_w_features[FEATURES])
        future_w_features['pred_req_vol'] = future_w_features['pred_req_vol'].astype(int)
        req_vol_interval = calculate_prediction_interval(future_w_features['pred_req_vol'])
        future_w_features['Req_Vol_Lower']=req_vol_interval[0].astype(int)
        future_w_features['Req_Vol_Upper']= req_vol_interval[1].astype(int)
        future_w_features['Req_Vol_Conf_score']=req_vol_interval[2].astype(int)
    else:
        print(f"model not found for volume for{sid}")


    if model_err_cnt is not None:
        future_w_features['pred_err_cnt'] = pred_err.predict(future_w_features[FEATURES])
        future_w_features['pred_err_cnt'] = future_w_features['pred_err_cnt'].astype(int)
        err_cnt_interval = calculate_prediction_interval(future_w_features['pred_err_cnt'])
        future_w_features['Err_Cnt_Lower']= err_cnt_interval[0]
        future_w_features['Err_Cnt_Upper']= err_cnt_interval[1]
        future_w_features['Err_Cnt_Conf_score']=err_cnt_interval[2]



    if model_resp_time is not None:
        future_w_features['pred_resp_time'] = pred_resp.predict(future_w_features[FEATURES])
        resp_time_interval = calculate_prediction_interval(future_w_features['pred_resp_time'])
        future_w_features['Resp_Time_Lower']= resp_time_interval[0]
        future_w_features['Resp_Time_Upper']= resp_time_interval[1]
        future_w_features['Resp_Time_Conf_score']=resp_time_interval[2]

    # future_w_features['Timestamp'] = future_w_features['datetime'].astype('int64') // 10**6

    # future_w_features['Timestamp'] = pd.to_datetime(future_w_features['Timestamp'], unit='ms')

    future_w_features['Timestamp']=future_w_features['datetime'].astype('int64')// 10**6
    future_w_features['fivemin_timestamp']=future_w_features['Timestamp']
    future_w_features['hour_timestamp'] = future_w_features['datetime'].dt.floor('H').astype('int64') // 10**6 # Hourly timestamp in milliseconds
    future_w_features['day_timestamp'] = future_w_features['datetime'].dt.floor('D').astype('int64')// 10**6  # Daily timestamp in milliseconds


    future_w_features['sid']=future_w_features['sid'].astype(str)


    print(f"printing the predicted values for {sid}")
    print(future_w_features)
    return future_w_features


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
        selected_columns = ['error_count', 'resp_time_sum','is_eb_breached','is_response_breached']

        # df_selected = df[['record_time', 'app_id', 'sid',]].join(df_extracted[selected_columns])

        df_selected = df[['record_time', 'app_id', 'sid','total_req_count']].join(df_extracted[selected_columns])

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
    # df_unique_services = pd.DataFrame({'sid': unique_services})

    # Sort the DataFrame by record_count in descending order
    df_unique_services = df_unique_services.sort_values(by='sid', ascending=True)

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


def remove_vol_outliers(df):
    # Remove outliers for req_vol
    Q1_req_vol = df['total_req_count'].quantile(0.15)
    Q3_req_vol = df['total_req_count'].quantile(0.95)
    IQR_req_vol = Q3_req_vol - Q1_req_vol

    outlier_mask_req_vol = (df['total_req_count'] < (Q1_req_vol - 1.5 * IQR_req_vol)) | (df['total_req_count'] > (Q3_req_vol + 1.5 * IQR_req_vol))
    df = df[~outlier_mask_req_vol]
    return df

def remove_resp_outliers(df):
    # Remove outliers for resp_time
    Q1_resp_time = df['resp_time_sum'].quantile(0.15)
    Q3_resp_time = df['resp_time_sum'].quantile(0.95)
    IQR_resp_time = Q3_resp_time - Q1_resp_time

    outlier_mask_resp_time = (df['resp_time_sum'] < (Q1_resp_time - 1.5 * IQR_resp_time)) | (df['resp_time_sum'] > (Q3_resp_time + 1.5 * IQR_resp_time))
    df = df[~outlier_mask_resp_time]
    return df


def remove_err_outliers(df):

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
def print_metrics(sid,values,metrics,inp):
    print(f'{metrics} for {sid}: ')
    print(f'Score across folds {inp}: {np.mean([score[0] for score in values]):0.4f}')
    print(f'Fold scores {inp}: {[score[0] for score in values]}')


def fit_req_vol_model(X_train, y_train_req_vol, X_test, y_test_req_vol, reg_req_vol):
    reg_req_vol.fit(X_train, y_train_req_vol,
                    eval_set=[(X_train, y_train_req_vol), (X_test, y_test_req_vol)],
                    verbose=100)

def fit_resp_time_model(X_train, y_train_resp_time, X_test, y_test_resp_time, reg_resp_time):
    reg_resp_time.fit(X_train, y_train_resp_time,
                      eval_set=[(X_train, y_train_resp_time), (X_test, y_test_resp_time)],
                      verbose=100)

def fit_err_cnt_model(X_train, y_train_err_cnt, X_test, y_test_err_cnt, reg_err_cnt):
    reg_err_cnt.fit(X_train, y_train_err_cnt,
                    eval_set=[(X_train, y_train_err_cnt), (X_test, y_test_err_cnt)],
                    verbose=100)

def mean_absolute_scale_error(y_true, y_pred):
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

def evaluate_metrics(y_test, pred):
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    # mase = mean_absolute_scale_error(y_test, pred)
    return rmse, mae, mape
# def evaluate_metrics_req_vol(y_test_req_vol, req_vol_pred):
#     req_vol_rmse = np.sqrt(mean_squared_error(y_test_req_vol, req_vol_pred))
#     req_vol_mae = mean_absolute_error(y_test_req_vol, req_vol_pred)
#     req_vol_mape = mean_absolute_percentage_error(y_test_req_vol, req_vol_pred)
#     req_vol_mase = mase(y_test_req_vol, req_vol_pred)
#     return req_vol_rmse, req_vol_mae, req_vol_mape, req_vol_mase
#
# def evaluate_metrics_resp_time(y_test_resp_time, resp_time_pred):
#     resp_time_rmse = np.sqrt(mean_squared_error(y_test_resp_time, resp_time_pred))
#     resp_time_mae = mean_absolute_error(y_test_resp_time, resp_time_pred)
#     resp_time_mape = mean_absolute_percentage_error(y_test_resp_time, resp_time_pred)
#     resp_time_mase = mase(y_test_resp_time, resp_time_pred)
#     return resp_time_rmse, resp_time_mae, resp_time_mape, resp_time_mase
#
# def evaluate_metrics_err_cnt(y_test_err_cnt, err_cnt_pred):
#     err_cnt_rmse = np.sqrt(mean_squared_error(y_test_err_cnt, err_cnt_pred))
#     err_cnt_mae = mean_absolute_error(y_test_err_cnt, err_cnt_pred)
#     err_cnt_mape = mean_absolute_percentage_error(y_test_err_cnt, err_cnt_pred)
#     err_cnt_mase = mase(y_test_err_cnt, err_cnt_pred)
#     return err_cnt_rmse, err_cnt_mae, err_cnt_mape, err_cnt_mase


# def predict_test(X_test,FEATURES,reg_req_vol,reg_resp_time,reg_err_cnt):
#     # X_test = pd.DataFrame(X_test, columns=FEATURES)  # Assuming FEATURES is a list of feature names
#
#     req_vol_pred = reg_req_vol.predict(X_test)
#     resp_time_pred = reg_resp_time.predict(X_test)
#     err_cnt_pred = reg_err_cnt.predict(X_test)
#
#     # # Calculate prediction intervals
#     # req_vol_interval = calculate_prediction_interval(X_train,y_train_req_vol,X_test)
#     # resp_time_interval = calculate_prediction_interval(X_train,y_train_resp_time,X_test)
#     # err_cnt_interval = calculate_prediction_interval(X_train,y_train_err_cnt,X_test)
#
#     # Calculate prediction intervals
#     req_vol_interval = calculate_prediction_interval(req_vol_pred)
#     resp_time_interval = calculate_prediction_interval(resp_time_pred)
#     err_cnt_interval = calculate_prediction_interval(err_cnt_pred)
#     return req_vol_pred,resp_time_pred,err_cnt_pred,req_vol_interval,resp_time_interval,err_cnt_interval

def predict_req_vol(reg_req_vol, X_test):
    req_vol_pred = reg_req_vol.predict(X_test)
    req_vol_interval = calculate_prediction_interval(req_vol_pred)
    return req_vol_pred,req_vol_interval

def predict_resp_time(reg_resp_time, X_test):
    resp_time_pred = reg_resp_time.predict(X_test)
    resp_time_interval = calculate_prediction_interval(resp_time_pred)
    return resp_time_pred,resp_time_interval

def predict_err_cnt(reg_err_cnt, X_test):
    err_cnt_pred = reg_err_cnt.predict(X_test)
    err_cnt_interval = calculate_prediction_interval(err_cnt_pred)
    return err_cnt_pred,err_cnt_interval
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



def save_csv(project_path,training_path,csv_name,df):
    csv_file_path =os.path.join(project_path, training_path, csv_name)  # Provide the desired file path
    df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to: {csv_file_path}")
    logging.info(f"DataFrame saved to: {csv_file_path}")


def train_for_service(filtered_df, sid, model_type):
    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df['record_time'] = df['record_time'].astype(np.int64) // 10**6  # Convert nanoseconds to milliseconds

    # start_time = timer()


    if len(filtered_df) <4:
        print(f"Not enough samples for {sid} time series cross-validation. Please add more data.")
    else:
        if model_type == 'vol':
            request_volume_model = train_volume_model_with_grid_search(filtered_df, sid)
            request_volume_model.save_model(os.path.join(project_path, models_path, f'{sid}_{model_type}.json'))
            print(f"Model saved for {sid} ({model_type})")
            return request_volume_model
        if model_type == 'resp':
            response_time_model = train_response_time_model_with_grid_search(filtered_df, sid)
            response_time_model.save_model(os.path.join(project_path, models_path, f'{sid}_{model_type}.json'))
            print(f"Model saved for {sid} ({model_type})")
            return response_time_model
        if model_type == 'err':
            error_count_model = train_error_count_model_with_grid_search(filtered_df, sid)
            error_count_model .save_model(os.path.join(project_path, models_path, f'{sid}_{model_type}.json'))
            print(f"Model saved for {sid} ({model_type})")
            return error_count_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}")




# @appp.route('/train')
# def train():
#     st_time=timer()
#     # To ignore specific warnings
#     warnings.filterwarnings("ignore", category=FutureWarning)
#
#     color_pal = sns.color_palette()
#     plt.style.use('fivethirtyeight')
#
#     # start_time=1701282600000
#     # end_time= 1701369000000
#     # start_time=1701408600000
#     # end_time= 1701495000000
#     # df=connectelk_retrive_data(app_id,start_time,end_time,source_index)
#     # save_csv(f'{source_index}.csv',df)
#     csv_file_path = os.path.join(project_path, training_path,'output_selected_dataframe.csv')
#     # csv_file_path = os.path.join(project_path, predictions_path,'1007_predicted.csv')
#
#     df = pd.read_csv(csv_file_path)
#
#
#     # bulk_save_forecast_index(df, target_index, elasticsearch_host, elasticsearch_port, elk_username,elk_password)
#     print(df.head())
#
#     logging.info(df.head())
#
#     correlation_threshold = 0.7
#
#     vec_data, venc_data, max_correlations_vol_err = classify_services(df,correlation_threshold,'vol_err')
#     vrc_data, vrnc_data, max_correlations_vol_resp = classify_services(df,correlation_threshold,'vol_resp')
#
#
#     df_corr_vol_err = list_unique_services(vec_data, max_correlations_vol_err)
#     df_corr_vol_resp= list_unique_services(vec_data, max_correlations_vol_resp)
#     df_corr_vol_err=df_corr_vol_err.sort_values(by='sid')
#     df_corr_vol_resp=df_corr_vol_resp.sort_values(by='sid')
#     save_csv( 'vol_err_corr_services.csv',df_corr_vol_err)# Provide the desired file path
#     save_csv( 'vol_resp_corr_services.csv',df_corr_vol_resp)# Provide the desired file path
#
#
#     df_all_services=update_unique_services(df)
#     # Sort the DataFrame by 'sid' in ascending order
#     df_all_services= df_all_services.sort_values(by='sid')
#     save_csv('all_services.csv',df_all_services)
#
#
#     # df_all_services = df_all_services['sid'].tolist()[:10]
#     df_all_services=[1007,1008,1021,1034,1036,1039]
#     # df_all_services = pd.DataFrame(df_all_services)
#     df_corr_vol_err = df_corr_vol_err['sid'].tolist()[:1]
#     print(df_corr_vol_err)
#     df_corr_vol_resp = df_corr_vol_resp['sid'].tolist()[:1]
#
#     all_forecasts = []
#     # for index, row in df_all_services.iterrows():
#     for sid in df_all_services[:1]:
#         # sid = row['sid']
#         df = df[df['sid'] == sid].sort_values(by='record_time')
#
#         df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')
#         save_csv(f'{sid}_train.csv', df)
#         df.set_index('record_time', inplace=True)
#
#         # train,test,df=train_test_splitdata(filtered_df)
#         df = create_features(df)
#         df = add_lags(df)
#
#         if sid in df_corr_vol_err:
#             # Train and predict for error
#             df_err=remove_err_outliers(df)
#             err_model = train_and_predict_for_service(df_err, sid, 'err')
#             # err_model=os.path.join(project_path, models_path,f'{sid}_err_cnt.json')
#
#         if sid in df_corr_vol_resp:
#             df_resp=remove_resp_outliers(df)
#             # Train and predict for response time
#             resp_model = train_and_predict_for_service(df_resp, sid, 'resp')
#             # resp_model=os.path.join(project_path, models_path,f'{sid}_resp_time.json')
#
#
#             # Train and predict for volume
#         df_vol=remove_vol_outliers(df)
#         vol_model = train_and_predict_for_service(df_vol, sid, 'vol')
#         # vol_model=os.path.join(project_path, models_path,f'{sid}_req_vol.json')
#
#
#         pred_start_date='2023-12-01 11:00:00'
#         pred_end_date='2023-12-20 11:00:00'
#         freq='5T'
#
#
#         forecast=predict_future_dataset(app_id,sid,freq,pred_start_date,pred_end_date)
#         print("saving to csv")
#         csv_file_path = os.path.join(project_path, predictions_path,f'{sid}_predicted.csv')  # Provide the desired file path
#         forecast.to_csv(csv_file_path, index=False)
#         print(f"Prediction DataFrame saved to: {csv_file_path}")
#
#     # def train_predict_for_service(sid):
#     #
#     #     # Train the model
#     #     filtered_df = df[df['sid'] == sid]
#     #     print(filtered_df.head())
#     #     filtered_df= filtered_df.sort_values(by='record_time')
#     #     save_csv(f'{sid}_train.csv',filtered_df)
#     #     train_and_predict(filtered_df,sid,pred_start_date,pred_end_date)
#     #
#     #  # Use ThreadPoolExecutor for parallel processing
#     # with ThreadPoolExecutor(max_workers=2) as executor:
#     #     executor.map(train_predict_for_service, services)
#     # ed_time=timer()
#     # total_execution_time = ed_time - st_time
#     #
#     # # Example response
#     response_data = {'status': 'success', 'message': 'Model trained successfully', 'total_execution_time': total_execution_time}
#     # response_data = {'status': 'success', 'message': 'prediction data saved successfully'}
#     logging.info(response_data)
#
#     # Return a valid HTTP response (JSON in this case)
#     return jsonify(response_data)

# @appp.route('/retrain')
# def retrain():
#     st_time=timer()
#     # To ignore specific warnings
#     warnings.filterwarnings("ignore", category=FutureWarning)
#
#     color_pal = sns.color_palette()
#     plt.style.use('fivethirtyeight')
#     # app_id="753"
#
#     # date=datetime.now() - timedelta(days=1)
#     # start_time = date.timestamp() * 1000  # Convert to milliseconds
#     # end_time = (date + timedelta(days=1)).timestamp() * 1000
#     start_time= 1700179200000
#     end_time=1701801000000
#     print(start_time)
#     print(end_time)
#     df = connectelk_retrive_data(app_id, start_time, end_time,source_index)
#
#     if df is not None:
#         print(df.head())
#
#         daily_unq_services=update_unique_services(df)
#         daily_unq_services= daily_unq_services.sort_values(by='sid')
#         csv_file_path = os.path.join(project_path,training_path,'daily_services.csv') # Provide the desired file path
#         daily_unq_services.to_csv(csv_file_path, index=False)
#         print(f"DataFrame saved to: {csv_file_path}")
#         logging.info(f"DataFrame saved to: {csv_file_path}")
#
#         services = daily_unq_services['sid'].tolist()[:5]
#         def retrain_model_for_service(sid):
#
#
#             # RETrain the model
#             filtered_df = df[df['sid'] == sid]
#             print(filtered_df.head())
#             filtered_df= filtered_df.sort_values(by='record_time')
#
#             csv_file_path = os.path.join(project_path, retraining_path,f'{sid}_retrain.csv')  # Provide the desired file path
#             filtered_df.to_csv(csv_file_path, index=False)
#             print(f"DataFrame saved to: {csv_file_path}")
#             logging.info(f"DataFrame saved to: {csv_file_path}")
#
#             if not filtered_df.empty:
#                 print(filtered_df.head())
#                 # check the model exists
#                 model_req_vol = os.path.join(project_path, models_path,f'{sid}_req_vol.json')  # Update with the actual file names
#                 model_resp_time = os.path.join(project_path, models_path,f'{sid}_resp_time.json')
#                 model_err_cnt = os.path.join(project_path, models_path,f'{sid}_err_cnt.json')
#                 if os.path.exists(model_req_vol) and os.path.exists(model_err_cnt) and os.path.exists(model_resp_time):
#                     model_req_vol, model_resp_time, model_err_cnt= grid_search_retrain_model(model_req_vol,
#                                                                                              model_err_cnt,
#                                                                                              model_resp_time,
#                                                                                              filtered_df, sid)
#                     # Save the updated model
#                     model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
#                     model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
#                     model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))
#
#                     pred_start_date='2023-12-11 11:00:00'
#                     pred_end_date='2023-12-12 11:00:00'
#                     freq='5T'
#                     forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
#                     # save_forecast_index(forecast,target_index)
#                     bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
#                                              elk_password)
#
#
#                 else:
#                     print(f"Model files not found for {sid}. Train new model first.")
#                     train_and_predict(filtered_df, sid)
#                     # model_req_vol, model_resp_time, model_err_cnt = train_and_forecast_service(filtered_df, sid)
#                     # model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
#                     # model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
#                     # model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))
#                     # if model_req_vol is not None:
#                     #
#                     #     pred_start_date='2023-12-11 11:00:00'
#                     #     pred_end_date='2023-12-22 11:00:00'
#                     #     freq='5T'
#                     #     forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
#                     #     # save_forecast_index(forecast,target_index)
#                     #     bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)
#
#             # all_services_df = pd.read_csv(os.path.join(project_path, training_path, 'all_services.csv'))
#             # all_services = all_services_df['sid']
#             #
#             # for sid in all_services[:3]:
#             #     if sid not in daily_services[:3]:  # Avoid predicting for services already processed during retraining
#             #         # Load or check the existence of the model
#             #         model_req_vol = os.path.join(project_path, models_path, f'{sid}_req_vol.json')
#             #         model_err_cnt = os.path.join(project_path, models_path, f'{sid}_err_cnt.json')
#             #         model_resp_time = os.path.join(project_path, models_path, f'{sid}_resp_time.json')
#             #
#             #         if os.path.exists(model_req_vol) and os.path.exists(model_err_cnt) and os.path.exists(model_resp_time):
#             #             # Predict the future data set
#             #             pred_start_date = '2023-11-22 11:00:00'
#             #             pred_end_date = '2023-12-11 11:00:00'
#             #             freq = '5T'
#             #             # filtered_df = connectelk_retrive_data(app_id, start_time, end_time)  # Adjust start_time and end_time as needed
#             #             forecast_future(app_id, pred_start_date, pred_end_date, freq,sid)
#             #         else:
#             #             print(f"Model files not found for {sid}. Train new model first.")
#             else:
#                 print("No records found for retraining model")
#         with ThreadPoolExecutor(max_workers=8) as executor:
#             executor.map(retrain_model_for_service, services)#
#         ed_time=timer()
#         total_execution_time = ed_time - st_time
#
#         # Example response
#         response_data = {'status': 'success', 'message': 'Model retrained successfully', 'total_execution_time': total_execution_time}
#         logging.info(response_data)
#
#         # Return a valid HTTP response (JSON in this case)
#         return jsonify(response_data)
#     else:
#         logging.error("Failed to load data.")
#         return None
# def train_request_volume_model(df,sid=None):
#         start_time = timer()
#         reg_req_vol, rmses, maes, mapes, mases= train_volume_model_with_grid_search(df)
#         print_metrics(sid,rmses,'RMSE','Request Volume')
#         print_metrics(sid,maes,'MAE','Request Volume')
#         print_metrics(sid,mapes,'MAPE','Request Volume')
#         print_metrics(sid,mases,'MASE','Request Volume')
#
#         # Print a message indicating the end of the training process
#
#         print(df.index.max())
#
#         end_time = timer()
#         training_time = end_time - start_time
#         print(f"Training completed for {sid} for Request Volume. Training time: {training_time} seconds.")
#         return reg_req_vol
    # Schedule the future prediction using the saved models
    # schedule.every().day.at("14:43").do(predict_future_dataset, model_req_vol, model_err_cnt, model_resp_time, service_name)

    # predict_future_dataset(model_req_vol,model_err_cnt,model_resp_time,service_name)
    # return model_req_vol,model_err_cnt,model_resp_time


def train_response_time_model(df,sid=None):

    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df['record_time'] = df['record_time'].astype(np.int64) // 10**6  # Convert nanoseconds to milliseconds
    start_time = timer()


    reg_resp_time, rmses, maes, mapes, mases = train_response_time_model_with_grid_search(df)
    print_metrics(sid,rmses,'RMSE','Response Time')
    print_metrics(sid,maes,'MAE','Response Time')
    print_metrics(sid,mapes,'MAPE','Response Time')
    print_metrics(sid,mases,'MASE','Response Time')

    # Print a message indicating the end of the training process

    print(df.index.max())

    end_time = timer()
    training_time = end_time - start_time
    print(f"Training completed for {sid} for Response time. Training time: {training_time} seconds.")
    return reg_resp_time
    # Schedule the future prediction using the saved models
    # schedule.every().day.at("14:43").do(predict_future_dataset, model_req_vol, model_err_cnt, model_resp_time, service_name)

    # predict_future_dataset(model_req_vol,model_err_cnt,model_resp_time,service_name)
    # return model_req_vol,model_err_cnt,model_resp_time

def train_error_count_model(df,sid=None):

    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df['record_time'] = df['record_time'].astype(np.int64) // 10**6  # Convert nanoseconds to milliseconds
    start_time = timer()


    reg_err_ent, rmses, maes, mapes, mases = train_error_count_model_with_grid_search(df)
    print_metrics(sid,maes,'MAE','Error Count')
    print_metrics(sid,mapes,'MAPE','Error Count')
    print_metrics(sid,mases,'MASE','Error Count')

    # Print a message indicating the end of the training process

    print(df.index.max())

    end_time = timer()
    training_time = end_time - start_time
    print(f"Training completed for {sid} for Error count. Training time: {training_time} seconds.")
    return reg_err_ent
    # Schedule the future prediction using the saved models
    # schedule.every().day.at("14:43").do(predict_future_dataset, model_req_vol, model_err_cnt, model_resp_time, service_name)

    # predict_future_dataset(model_req_vol,model_err_cnt,model_resp_time,service_name)
    # return model_req_vol,model_err_cnt,model_resp_time




def grid_search_retrain_model(model_req_vol, model_err_cnt, model_resp_time, df,sid):


    df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')

    df.set_index('record_time', inplace=True)

    # Set the index to the 'datetime' column
    #     df.set_index('datetime', inplace=True)
    # df=remove_outliers(df)
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

# def save_csv_to_opensearch(csv_path, index_name, es_host, es_port, es_username, es_password):
#     # Read CSV into a Pandas DataFrame
#     df = pd.read_csv(csv_path)
#
#     # Convert DataFrame to a list of dictionaries (each dictionary represents a document)
#     documents = df.to_dict(orient='records')
#
#     # Create an Elasticsearch connection
#     es = Elasticsearch([{'host': es_host, 'port': es_port}], http_auth=(es_username, es_password))
#
#     # Bulk index the documents into the specified OpenSearch index
#     try:
#         success, failed = bulk(es, documents, index=index_name, raise_on_error=True)
#         print(f"Successfully indexed {success} documents into '{index_name}' index.")
#         if failed:
#             print(f"Failed to index {failed} documents. Check for errors.")
#     except Exception as e:
#         print(f"Error bulk indexing data: {e}")
import http.client
import base64
def delete_open_search_index(index_name, host, port, username, password):
    connection = http.client.HTTPConnection(host, port)
    url = f'/{index_name}'

    try:
        connection.request('DELETE', url, headers={'Authorization': f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'})
        response = connection.getresponse()

        if response.status == 200:
            print(f"Index '{index_name}' deleted successfully.")
        else:
            print(f"Failed to delete index '{index_name}'. Status code: {response.status}")

    except Exception as e:
        print(f"Error deleting index '{index_name}': {e}")

    finally:
        connection.close()
    # url = f"http://{host}:{port}/{index_name}"
    # auth = (username, password)
    #
    # try:
    #     response = requests.delete(url, auth=auth)
    #     response.raise_for_status()  # Check for HTTP errors
    #
    #     if response.status_code == 200:
    #         print(f"Index '{index_name}' deleted successfully.")
    #     else:
    #         print(f"Failed to delete index '{index_name}'. Status code: {response.status_code}")
    #
    # except requests.exceptions.RequestException as e:
    #     print(f"Error deleting index '{index_name}': {e}")

# Example usage:
def list_services_training(df):
    correlation_threshold = 0.7

    vec_data, venc_data, max_correlations_vol_err = classify_services(df,correlation_threshold,'vol_err')
    vrc_data, vrnc_data, max_correlations_vol_resp = classify_services(df,correlation_threshold,'vol_resp')


    df_corr_vol_err = list_unique_services(vec_data, max_correlations_vol_err)
    df_corr_vol_resp= list_unique_services(vec_data, max_correlations_vol_resp)
    df_corr_vol_err=df_corr_vol_err.sort_values(by='sid')
    df_corr_vol_resp=df_corr_vol_resp.sort_values(by='sid')
    save_csv(project_path,training_path, 'vol_err_corr_services.csv',df_corr_vol_err)# Provide the desired file path
    save_csv(project_path,training_path, 'vol_resp_corr_services.csv',df_corr_vol_resp)# Provide the desired file path


    df_all_services=update_unique_services(df)
    # Sort the DataFrame by 'sid' in ascending order
    df_all_services= df_all_services.sort_values(by='sid')
    save_csv(project_path,training_path,'all_services.csv',df_all_services)
    return df_all_services,df_corr_vol_err,df_corr_vol_resp

def filter_service_by_sid(df,sid):
    df_service = df[df['sid'] == sid].sort_values(by='record_time')

    df_service['record_time'] = pd.to_datetime(df_service['record_time'], unit='ms')
    save_csv(project_path,training_path,f'{sid}_train.csv', df_service)
    df_service.set_index('record_time', inplace=True)
    return df_service
def list_services_retraining(df):
    correlation_threshold = 0.7

    vec_data, venc_data, max_correlations_vol_err = classify_services(df,correlation_threshold,'vol_err')
    vrc_data, vrnc_data, max_correlations_vol_resp = classify_services(df,correlation_threshold,'vol_resp')


    df_corr_vol_err = list_unique_services(vec_data, max_correlations_vol_err)
    df_corr_vol_resp= list_unique_services(vec_data, max_correlations_vol_resp)
    df_corr_vol_err=df_corr_vol_err.sort_values(by='sid')
    df_corr_vol_resp=df_corr_vol_resp.sort_values(by='sid')
    save_csv(project_path,retraining_path, 'vol_err_corr_services.csv',df_corr_vol_err)# Provide the desired file path
    save_csv(project_path,retraining_path, 'vol_resp_corr_services.csv',df_corr_vol_resp)# Provide the desired file path


    df_all_services=update_unique_services(df)
    # Sort the DataFrame by 'sid' in ascending order
    df_all_services= df_all_services.sort_values(by='sid')
    save_csv(project_path,retraining_path,'all_services.csv',df_all_services)
    return df_all_services,df_corr_vol_err,df_corr_vol_resp
def retrain_for_service(filtered_df, sid, model_type):
    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df['record_time'] = df['record_time'].astype(np.int64) // 10**6  # Convert nanoseconds to milliseconds

    # start_time = timer()


    if len(filtered_df) <4:
        print(f"Not enough samples for {sid} time series cross-validation. Please add more data.")
    else:
        if model_type == 'vol':
            request_volume_model = retrain_request_volume_model(filtered_df, sid)
            return request_volume_model
        if model_type == 'resp':
            response_time_model = retrain_response_time_model(filtered_df, sid)
            return response_time_model
        if model_type == 'err':
            error_count_model = retrain_error_count_model(filtered_df, sid)
            return error_count_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

def retrain_request_volume_model(df,sid):
    print(f"Retraining completed for {sid} for Request Volume")
    start_time = timer()
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes=[]
        mapes=[]
        mases=[]

        params = {
            'objective': 'reg:squarederror',
            'base_score':0.5, 'booster':'gbtree',
            'n_estimators':800,
            'early_stopping_rounds':50,
            'max_depth':15,
            'learning_rate':0.2
        }
        model_req_vol_path = os.path.join(project_path, models_path, f'{sid}_vol.json')

        if os.path.exists(model_req_vol_path):
            model_req_vol = xgb.Booster(model_file=model_req_vol_path)

        else:
            model_req_vol = XGBRegressor(**params)

        for train_idx, val_idx in tss.split(df):

            req_vol_preds = []

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
            # 'lag1','lag2','lag3']
            REQ_VOL_TARGET = 'total_req_count'

            # Add lags and create features
            df = add_lags(df)  # Assuming you have a function to add lags
            df = create_features(df)

            new_X_req_vol = test[FEATURES]
            new_y_req_vol = test[REQ_VOL_TARGET]

            dnew_req_vol = xgb.DMatrix(new_X_req_vol, label=new_y_req_vol)

            model_req_vol = xgb.train(params, dnew_req_vol, num_boost_round=30, xgb_model=model_req_vol)


            # Make predictions with the updated model
            # req_vol_pred = model_req_vol.predict(new_X_req_vol)
            req_vol_pred = model_req_vol.predict(dnew_req_vol)

            # req_vol_pred, req_vol_interval = predict_req_vol( updated_model_req_vol,new_X_req_vol)
            # Append intervals to your lists
            # req_vol_preds.append((req_vol_pred, req_vol_interval))

            req_vol_rmse, req_vol_mae, req_vol_mape, req_vol_mase=evaluate_metrics(new_y_req_vol, req_vol_pred)

            rmses.append((req_vol_rmse,))
            maes.append((req_vol_mae,))
            mapes.append((req_vol_mape,))
            mases.append((req_vol_mase,))

        print_metrics(sid,rmses,'RMSE','Request Volume')
        print_metrics(sid,maes,'MAE','Request Volume')
        print_metrics(sid,mapes,'MAPE','Request Volume')
        print_metrics(sid,mases,'MASE','Request Volume')

        # Save the final model
        model_req_vol.save_model(model_req_vol_path)
        print(f"Request Volume Model saved for {sid} in f'{sid}_vol.json' ")

        end_time = timer()
        training_time = end_time - start_time
        print(f"Retraining completed for {sid} for Request Volume. Training time: {training_time} seconds.")

        # Return the final model
        return model_req_vol

    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service {sid} due to error: {e}")


    return None
def retrain_response_time_model(df,sid):
    print(f"Retraining completed for {sid} for REsponse Time")
    start_time = timer()
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes=[]
        mapes=[]
        mases=[]

        params = {
            'objective': 'reg:squarederror',
            'base_score':0.5, 'booster':'gbtree',
            'n_estimators':1000,
            'early_stopping_rounds':50,
            'max_depth':20,
            'learning_rate':0.01
        }
        model_resp_time_path = os.path.join(project_path, models_path, f'{sid}_resp.json')

        if os.path.exists(model_resp_time_path):
            model_resp_time = xgb.Booster(model_file=model_resp_time_path)

        else:
            model_resp_time = XGBRegressor(**params)

        for train_idx, val_idx in tss.split(df):

            resp_time_preds = []

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
            # 'lag1','lag2','lag3']
            RESP_TIME_TARGET = 'resp_time_sum'

            # Add lags and create features
            df = add_lags(df)  # Assuming you have a function to add lags
            df = create_features(df)

            new_X_resp_time = test[FEATURES]
            new_y_resp_time = test[RESP_TIME_TARGET]

            dnew_resp_time = xgb.DMatrix(new_X_resp_time, label=new_y_resp_time)

            model_resp_time = xgb.train(params, dnew_resp_time, num_boost_round=30, xgb_model=model_resp_time)


            # Make predictions with the updated model
            # resp_time_pred = model_resp_time.predict(new_X_resp_time)
            resp_time_pred = model_resp_time.predict(dnew_resp_time)

            # resp_time_pred, resp_time_interval = predict_resp_time( updated_model_resp_time,new_X_resp_time)
            # Append intervals to your lists
            # resp_time_preds.append((resp_time_pred, resp_time_interval))

            resp_time_rmse, resp_time_mae, resp_time_mape, resp_time_mase=evaluate_metrics(new_y_resp_time, resp_time_pred)

            rmses.append((resp_time_rmse,))
            maes.append((resp_time_mae,))
            mapes.append((resp_time_mape,))
            mases.append((resp_time_mase,))

        print_metrics(sid,rmses,'RMSE','Request Volume')
        print_metrics(sid,maes,'MAE','Request Volume')
        print_metrics(sid,mapes,'MAPE','Request Volume')
        print_metrics(sid,mases,'MASE','Request Volume')

        # Save the final model
        model_resp_time.save_model(model_resp_time_path)
        print(f"Request Volume Model saved for {sid} in f'{sid}_vol.json' ")

        end_time = timer()
        training_time = end_time - start_time
        print(f"Retraining completed for {sid} for Request Volume. Training time: {training_time} seconds.")

        # Return the final model
        return model_resp_time

    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service {sid} due to error: {e}")


    return None
def retrain_error_count_model(df,sid):
    print(f"Retraining completed for {sid} for Error Count")
    start_time = timer()
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes=[]
        mapes=[]
        mases=[]

        params = {
            'objective': 'reg:squarederror',
            'base_score':0.5, 'booster':'gbtree',
            'n_estimators':1000,
            'early_stopping_rounds':50,
            'max_depth':20,
            'learning_rate':0.01
        }
        model_err_cnt_path = os.path.join(project_path, models_path, f'{sid}_resp.json')

        if os.path.exists(model_err_cnt_path):
            model_err_cnt = xgb.Booster(model_file=model_err_cnt_path)

        else:
            model_err_cnt = XGBRegressor(**params)

        for train_idx, val_idx in tss.split(df):

            err_cnt_preds = []

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
            # 'lag1','lag2','lag3']
            ERR_CNT_TARGET = 'err_cnt_sum'

            # Add lags and create features
            df = add_lags(df)  # Assuming you have a function to add lags
            df = create_features(df)

            new_X_err_cnt = test[FEATURES]
            new_y_err_cnt = test[ERR_CNT_TARGET]

            dnew_err_cnt = xgb.DMatrix(new_X_err_cnt, label=new_y_err_cnt)

            model_err_cnt = xgb.train(params, dnew_err_cnt, num_boost_round=30, xgb_model=model_err_cnt)


            # Make predictions with the updated model
            # err_cnt_pred = model_err_cnt.predict(new_X_err_cnt)
            err_cnt_pred = model_err_cnt.predict(dnew_err_cnt)

            # err_cnt_pred, err_cnt_interval = predict_err_cnt( updated_model_err_cnt,new_X_err_cnt)
            # Append intervals to your lists
            # err_cnt_preds.append((err_cnt_pred, err_cnt_interval))

            err_cnt_rmse, err_cnt_mae, err_cnt_mape, err_cnt_mase=evaluate_metrics(new_y_err_cnt, err_cnt_pred)

            rmses.append((err_cnt_rmse,))
            maes.append((err_cnt_mae,))
            mapes.append((err_cnt_mape,))
            mases.append((err_cnt_mase,))

        print_metrics(sid,rmses,'RMSE','Request Volume')
        print_metrics(sid,maes,'MAE','Request Volume')
        print_metrics(sid,mapes,'MAPE','Request Volume')
        print_metrics(sid,mases,'MASE','Request Volume')

        # Save the final model
        model_err_cnt.save_model(model_err_cnt_path)
        print(f"Request Volume Model saved for {sid} in f'{sid}_vol.json' ")

        end_time = timer()
        training_time = end_time - start_time
        print(f"Retraining completed for {sid} for Request Volume. Training time: {training_time} seconds.")

        # Return the final model
        return model_err_cnt

    except ValueError as e:
        # Handle the exception (e.g., print a message)
        print(f"Skipping service{sid} due to error: {e}")


    return None
from concurrent.futures import ThreadPoolExecutor

def pull_training_data(app_id,source_index):
    # start_time_ms=1701408600000
    # end_time_ms= 1701495000000

    current_date=datetime.now()
    # Set start time to 00:00 hour of the previous day
    start_time = current_date - timedelta(days=9)
    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time_ms = int(start_time.timestamp() * 1000)


# Set end time to 23:59 of the previous day
    end_time = datetime.now() - timedelta(days=1)
    end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=0)
    end_time_ms = int(end_time.timestamp() * 1000)




    df=connectelk_retrive_data(app_id,start_time_ms,end_time_ms,source_index)

    csv_file_path = os.path.join(project_path, training_path, f'training_data.csv')
    # df = pd.read_csv(csv_file_path)
    df.to_csv(csv_file_path, index=False)
    print(f"Tranining DataFrame saved to: {csv_file_path}")

    return df

def get_pred_date():
    current_date=datetime.now()
    days = int(config['prediction']['days'])


    # Calculate the prediction start date (10 days ahead)
    prediction_start_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
    prediction_end_date = prediction_start_date + timedelta(days=days) - timedelta(seconds=1)

    # Convert the dates to strings
    pred_start_date = prediction_start_date.strftime('%Y-%m-%d %H:%M:%S')
    pred_end_date = prediction_end_date.strftime('%Y-%m-%d %H:%M:%S')

    print("Prediction Start Date:", pred_start_date)
    print("Prediction End Date:", pred_end_date)
    return pred_start_date,pred_end_date
@appp.route('/train')
def train():
    st_time = timer()
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')


    # df=pull_training_data()
    file_path = r'C:\Users\91701\EB_ML\fivemins_wm_oseb1decv1_753.csv'
    # csv_file_path = os.path.join(project_path, training_path, 'output_selected_dataframe.csv')
    # csv_file_path = os.path.join(project_path, '..', 'EB_ML', 'training', 'output_selected_dataframe.csv')

    df = pd.read_csv(file_path)

    logging.info(df.head())

    df_all_services, df_corr_vol_err, df_corr_vol_resp = list_services_training(df)

    df_all_services = df_all_services['sid'].tolist()[:20]
    print('df_All_services in training', df_all_services)

    # df_all_services = [1007, 1008, 1021, 1034]
    df_corr_vol_err = df_corr_vol_err['sid'].tolist()
    df_corr_vol_resp = df_corr_vol_resp['sid'].tolist()
    print('vol and err correlation list in training', df_corr_vol_err)

    # Step 1: Model Training
    def train_models(sid):
        try:
            df_service = filter_service_by_sid(df, sid)

            if sid in df_corr_vol_err:
                df_err = remove_err_outliers(df_service)
                err_model = train_for_service(df_err, sid, 'err')

            if sid in df_corr_vol_resp:
                df_resp = remove_resp_outliers(df_service)
                resp_model = train_for_service(df_resp, sid, 'resp')

            df_vol = remove_vol_outliers(df_service)
            vol_model = train_for_service(df_vol, sid, 'vol')
        except Exception as e:
            print(f"An exception occurred during training for service {sid}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(train_models, df_all_services)

    # Step 2: Prediction
    all_forecasts = []

    def predict(sid):
        try:

            # pred_start_date,pred_end_date= get_pred_date()
            pred_start_date='2023-12-05 00:00:00'
            pred_end_date='2023-12-07 23:59:59'


            freq = '5T'

            forecast = predict_future_dataset(app_id, sid, freq, pred_start_date, pred_end_date)

            csv_file_path = os.path.join(project_path, predictions_path, f'{sid}_predicted.csv')
            forecast.to_csv(csv_file_path, index=False)
            print(f"Prediction DataFrame saved to: {csv_file_path}")

            # bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

            all_forecasts.append(forecast)

        except Exception as e:
            print(f"An exception occurred during prediction for service {sid}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(predict, df_all_services)

    total_execution_time = timer() - st_time
    #
    # # Concatenate all forecasts into a single DataFrame
    # combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    #
    # # Save the combined forecasts to a CSV file
    # combined_csv_path = os.path.join(project_path, predictions_path, 'combined_forecasts.csv')
    # combined_forecasts.to_csv(combined_csv_path, index=False)
    # print(f"Combined Forecast CSV saved to: {combined_csv_path}")

    response_data = {'status': 'success', 'message': 'Model trained and prediction saved successfully',
                     'total_execution_time': total_execution_time}
    logging.info(response_data)

    return jsonify(response_data)

def get_retraining_start_end_time():
    current_date=datetime.now()
    # Set start time to 00:00 hour of the previous day
    start_time = current_date - timedelta(days=1)
    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Set end time to 23:59 of the previous day
    end_time = current_date - timedelta(days=1)
    end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=0)

    current_timestamp_ms = int(current_date.timestamp() * 1000)

    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    print("Current time:", current_timestamp_ms)
    print("Start time:", start_time_ms)
    print("End time:", end_time_ms)

    return start_time_ms,end_time_ms

@appp.route('/retrain')

def retrain(app):
    logging.info(f"using applicationid:{app.id}")
    st_time=timer()
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')
    app_id=app.id
    # app_id='753'
    source_index = 'fivemins_{}'.format(app.index_id)
    # source_index= 'fivemins_wm_oseb1decv1_753'

    print(app_id,source_index)

    # csv_file_path = os.path.join(project_path, training_path, 'output_selected_dataframe.csv')
    #
    # df_train_data = pd.read_csv(csv_file_path)
    df_train_data = pull_training_data(app_id, source_index) # fetch 90 days data


    #
    # start_time= 1701648000000 # Dec 4,2023 12 am
    # end_time=1701734400000 #Dec 5,2023 12 am
    #

    # fetch Retrain data last 1 day
    start_time_ms,end_time_ms=get_retraining_start_end_time()
    start_time_ms=1706034600000 # REMOVE THIS
    end_time_ms=1706045400000 # REMOVE THIS
    df = connectelk_retrive_data(app_id, start_time_ms, end_time_ms,source_index) # fetch last 1 day

    csv_file_path = os.path.join(project_path, retraining_path, 'retraining_data.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Retranining DataFrame saved to: {csv_file_path}")
    # df = pd.read_csv(csv_file_path)

    if df is not None:
        df['fivemin_timestamp']=df['record_time']
        # Assuming your DataFrame is named 'df'
        df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')
        # Extract hour and day from the 'record_time'
        df['hour_timestamp'] = df['record_time'].dt.floor('H').astype('int64') // 10**6 # Hourly timestamp in milliseconds
        df['day_timestamp'] = (df['record_time'].dt.floor('D')).astype('int64') // 10**6 # Daily timestamp in milliseconds


        # bulk_save_training_index(df, training_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)
        print(df.head())

        logging.info(df.head())

        df_all_services, df_corr_vol_err, df_corr_vol_resp = list_services_training(df)

        df_all_services = df_all_services['sid'].tolist()[:10]
        print('df_All_services in retraining', df_all_services)

        # df_all_services = [1007, 1008, 1021, 1034]
        df_corr_vol_err = df_corr_vol_err['sid'].tolist()
        df_corr_vol_resp = df_corr_vol_resp['sid'].tolist()
        print('vol and err correlation list in retraining', df_corr_vol_err)



        # Step 1: Model Retraining
        def retrain_model_for_service(sid):

            try:
                df_service  =filter_service_by_sid( df,sid)

                if not df_service.empty:
                    model_req_vol = os.path.join(project_path, models_path,f'{sid}_vol.json')  # Update with the actual file names
                    model_resp_time = os.path.join(project_path, models_path,f'{sid}_resp.json')
                    model_err_cnt = os.path.join(project_path, models_path,f'{sid}_err.json')
                    if sid in df_corr_vol_err:
                        if os.path.exists(model_err_cnt):
                            # Train and predict for error
                            df_err=remove_err_outliers(df_service)
                            err_model = retrain_for_service(df_err, sid, 'err')
                        else:

                            df_service  =filter_service_by_sid(df_train_data,sid)
                            df_err=remove_err_outliers(df_service)
                            print(f"No existing Error count Model for {sid} to retrain .So training the model for Error Count ")
                            train_for_service(df_err, sid, 'err')

                    if sid in df_corr_vol_resp:
                        if os.path.exists(model_resp_time):

                            df_resp=remove_resp_outliers(df_service)
                            # Train and predict for response time
                            resp_model = retrain_for_service(df_resp, sid, 'resp')
                        else:
                            df_service  =filter_service_by_sid(df_train_data,sid)
                            df_resp=remove_resp_outliers(df_service)
                            print(f"No existing Response Time Model for {sid} to retrain .So training the model for Response time  ")

                            train_for_service(df_resp, sid, 'resp')

                    if os.path.exists(model_req_vol):
                        # Train and predict for volume
                        df_vol=remove_vol_outliers(df_service)
                        vol_model = retrain_for_service(df_vol, sid, 'vol')
                    else:
                        df_service  =filter_service_by_sid(df_train_data,sid)
                        df_vol=remove_vol_outliers(df_service)
                        print(f"No existing Volume Model for {sid} to retrain .so Training the model for Volume  ")

                        train_for_service(df_vol, sid, 'vol')

            except Exception as e:
                print(f"An exception occurred during training for service {sid}: {e}")
        with ThreadPoolExecutor() as executor:
            executor.map(retrain_model_for_service, df_all_services)
    else:
        print(f"No records found for  retraining model")

    # Step 2: Prediction
    all_forecasts = []
    # # Delete the old target OpenSearch index
    # delete_open_search_index(target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)
    all_tomrw_forecasts=[]
    def predict(sid):
        try:

            pred_start_date,pred_end_date= get_pred_date()

            # pred_start_date='2023-12-05 00:00:00'
            # pred_end_date='2023-12-07 23:59:59'

            freq='5T'


            forecast=predict_future_dataset(app_id,sid,freq,pred_start_date,pred_end_date,df_corr_vol_err,df_corr_vol_resp)

            # df_service  =filter_service_by_sid( df,sid)
            # plot_predictions(df_service,forecast,sid,freq)

            all_forecasts.append(forecast)

            csv_file_path = os.path.join(project_path, predictions_path,f'{sid}_predicted.csv')  # Provide the desired file path
            forecast.to_csv(csv_file_path, index=False)
            print(f"Prediction DataFrame saved to: {csv_file_path}")


            bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,elk_password)

            print(f"retrieving tomorrow data for {sid}")

            # print(f"Type of forecast['datetime'] column: {type(forecast['datetime'])}")

            # Convert the string to a datetime object
            pred_start_date = datetime.strptime(pred_start_date, '%Y-%m-%d %H:%M:%S')
            # print(f"Type of pred_start_date: {type(pred_start_date)}")

            # Now try to retrieve tomorrow's predictions
            try:
                # Convert the 'datetime' column to dates
                forecast['date'] = forecast['datetime'].dt.date

                # Convert the pred_start_date to a date
                pred_start_date = pred_start_date.date()

                # Filter the DataFrame for tomorrow's predictions
                tomorrow_predictions = forecast[forecast['date'] == pred_start_date]

                all_tomrw_forecasts.append(tomorrow_predictions)

                print(f"Tomorrow's predictions: {tomorrow_predictions}")

            except Exception as e:
                print(f"An exception occurred during filtering of tomorrow prediction for service {sid}: {e}")


        except Exception as e:
            print(f"An exception occurred during prediction for service {sid}: {e}")


    with ThreadPoolExecutor() as executor:
        executor.map(predict, df_all_services)

    # Reset the index of df and tomorrow_predictions
    df.reset_index(drop=True)
    all_tomrw_forecasts_df = pd.concat(all_tomrw_forecasts, ignore_index=True)
    all_tomrw_forecasts_df.reset_index(drop=True)

    # Concatenate df and tomorrow_predictions along the rows without preserving the index
    combined_df = pd.concat([df, all_tomrw_forecasts_df], ignore_index=True)

    bulk_save_forecast_index(combined_df, training_index , elasticsearch_host, elasticsearch_port, elk_username,elk_password)

    # Save the combined DataFrame to the CSV file with a date index
    combined_csv_file_path = os.path.join(project_path, predictions_path, 'training_day_pred.csv')
    combined_df.to_csv(combined_csv_file_path, index=False)
    print(f"Combined Data (retraining and one day prediction )Saved to: {combined_csv_file_path}")



    total_execution_time = timer() - st_time
    #
    # # Concatenate all forecasts into a single DataFrame
    # combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    #
    # # Save the combined forecasts to a CSV file
    # combined_csv_path = os.path.join(project_path, predictions_path, 'retrained_combined_forecasts.csv')
    # combined_forecasts.to_csv(combined_csv_path, index=False)
    # print(f"Combined Forecast CSV saved to: {combined_csv_path}")

    # Delete the old target OpenSearch index
    # delete_open_search_index(target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

    # Save the combined forecasts to OpenSearch
    # bulk_save_forecast_index(combined_forecasts, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

    # Example response
    response_data = {'status': 'success', 'message': 'Model retrained successfully', 'total_execution_time': total_execution_time}
    logging.info(response_data)

    # Return a valid HTTP response (JSON in this case)
    return jsonify(response_data)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train or retrain the model.')
    # parser.add_argument('mode', choices=['train', 'retrain'], default='train', help='Specify train or retrain mode')
    #
    # args = parser.parse_args()
    # main(args.mode)
    appp.run(debug=True)



