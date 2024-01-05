import logging
from ap import connectelk_retrive_data,update_unique_services,train_and_predict,grid_search_retrain_model
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
from concurrent.futures import ThreadPoolExecutor
from functools import partial

app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Get the current working directory
project_path = os.getcwd()

# Specify the relative path to your models directory
models_path = 'models'
predictions_path='predictions'
training_path='training'
retraining_path='retraining'

app_id="753"
source_index='daily_wm_oseb1decv1_753'
target_index='test_ai_dheepa_13dec_01'

@app.route('/')
def index():
    return 'Welcome to your EB_ML Flask App!'
# Common function for loading and processing data
def load_and_process_data(start_time, end_time, app_id, source_index, project_path, training_path):
    df = connectelk_retrive_data(app_id, start_time, end_time, source_index)
    if df is not None:
        df_unique_services = update_unique_services(df)
        df_unique_services = df_unique_services.sort_values(by='sid')
        csv_file_path = os.path.join(project_path, training_path, 'all_services.csv')
        df_unique_services.to_csv(csv_file_path, index=False)
        logging.info(f"DataFrame saved to: {csv_file_path}")
        return df_unique_services
    else:
        logging.error("Failed to load data.")
        return None

# Parallel processing function
def parallel_process_services(services, process_function, max_workers=8, **kwargs):
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(process_function, services, **kwargs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use functools.partial to pass additional arguments to process_function
        partial_process_function = partial(process_function, **kwargs)
        executor.map(partial_process_function, services)


# Function to check the existence of model files
def check_model_files_exist(model_paths):
    return all(os.path.exists(model_path) for model_path in model_paths)

# Common function for training and retraining a model for a service
def train_or_retrain_model(sid, df, project_path, models_path, training_function, retraining_function):
    filtered_df = df[df['sid'] == sid]
    filtered_df = filtered_df.sort_values(by='record_time')

    csv_file_path = os.path.join(project_path, training_path, f'{sid}_train.csv')
    filtered_df.to_csv(csv_file_path, index=False)
    logging.info(f"DataFrame saved to: {csv_file_path}")

    # Update with the actual file names
    model_req_vol = os.path.join(project_path, models_path, f'{sid}_req_vol.json')
    model_resp_time = os.path.join(project_path, models_path, f'{sid}_resp_time.json')
    model_err_cnt = os.path.join(project_path, models_path, f'{sid}_err_cnt.json')

    if check_model_files_exist([model_req_vol, model_err_cnt, model_resp_time]):
        retraining_function(model_req_vol, model_err_cnt, model_resp_time, filtered_df, sid)
    else:
        training_function(filtered_df, sid)

# Training function
def train_model_for_service(sid, df, project_path, models_path, training_function):
    train_or_retrain_model(sid, df, project_path, models_path, training_function, train_and_predict)

# Retraining function
def retrain_model_for_service(sid, df, project_path, models_path, retraining_function):
    train_or_retrain_model(sid, df, project_path, models_path, retraining_function, grid_search_retrain_model)

@app.route('/train')
def train():
    st_time = timer()
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')

    start_time = 1699900200000
    end_time = 1701541800000

    df_unique_services = load_and_process_data(start_time, end_time, app_id, source_index, project_path, training_path)

    if df_unique_services is not None:
        services = df_unique_services['sid'].tolist()[:2]
        parallel_process_services(services, train_model_for_service, max_workers=8,
                                  df=df_unique_services, project_path=project_path, models_path=models_path)

    ed_time = timer()
    total_execution_time = ed_time - st_time

    response_data = {'status': 'success', 'message': 'Model trained successfully',
                     'total_execution_time': total_execution_time}
    return jsonify(response_data)

@app.route('/retrain')
def retrain():
    st_time = timer()
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')

    start_time = 1700179200000
    end_time = 1701801000000

    df = load_and_process_data(start_time, end_time, app_id, source_index, project_path, training_path)

    if df is not None:
        services = df['sid'].tolist()[:2]
        parallel_process_services(services, retrain_model_for_service, max_workers=8,
                                  df=df, project_path=project_path, models_path=models_path)

    ed_time = timer()
    total_execution_time = ed_time - st_time

    response_data = {'status': 'success', 'message': 'Model retrained successfully',
                     'total_execution_time': total_execution_time}
    return jsonify(response_data)

if __name__ == "__main__":

    app.run(debug=True)