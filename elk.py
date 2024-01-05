import requests
import certifi
from requests.auth import HTTPBasicAuth

from elasticsearch.helpers import bulk
# from elasticsearch.exceptions import ElasticsearchException
from XGB_functions import train_model_with_grid_search,save_forecast_index, predict_future_dataset,plot_predictions, prediction_input,connect_db,add_lags,create_features,calculate_prediction_interval
from elasticsearch import Elasticsearch
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit,GridSearchCV

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from XGB_functions import connectelk_retrive_data ,list_unique_services, classify_services,update_unique_services,connect_db,add_lags,create_features,calculate_prediction_interval,remove_outliers,train_test_splitdata,train_model,fit_model,predict_test,print_metrics,evaluate_metrics
import schedule
import argparse
import pickle
import os
from timeit import default_timer as timer
from datetime import datetime, timedelta
# Get the current working directory
project_path = os.getcwd()

# Specify the relative path to your models directory
models_path = 'models'
predictions_path='predictions'
training_path='training'
retraining_path='retraining'




def train_and_forecast_service(df,sid=None):

    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df['record_time'] = df['record_time'].astype(np.int64) // 10**6  # Convert nanoseconds to milliseconds
    start_time = timer()

    df['record_time'] = pd.to_datetime(df['record_time'], unit='ms')

    df.set_index('record_time', inplace=True)

    # df = df.set_index('datetime')

    # df.index = pd.to_datetime(df.index)

    # df.plot(style='.',
    #         figsize=(15, 5),
    #         # color=color_pal[0],
    #         title='Request volume')
    # plt.show()
    # df['req_vol'].plot(kind='hist', bins=500)
    # plt.show()
    #
    # df.query('req_vol >500000')['req_vol'] \
    #         .plot(style='.',
    #               figsize=(15, 5),
    #               # color=color_pal[5],
    #               title='Outliers')
    # plt.show()

    # df = df.query('req_vol <=500000').copy()


    df=remove_outliers(df)
    train,test,df=train_test_splitdata(df)

    # from scipy import stats
    #
    # z_scores = stats.zscore(df['req_vol'])
    # abs_z_scores = np.abs(z_scores)
    # outlier_mask = (abs_z_scores > 3)
    # df= df[~outlier_mask]


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
def retrainn_model(model_req_vol, model_err_cnt, model_resp_time, df,sid):
    # Assuming df is the new data for retraining

    # Convert 'datetime' column to datetime object
    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df = df.set_index('datetime')
    # df.index = pd.to_datetime(df.index, format="%b %d, %Y @ %H:%M:%S.%f")

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

# # You can further evaluate the performance on a test set or do additional checks
        # req_vol_rmse = mean_squared_error(test['req_vol'], pred_req_vol, squared=False)
        # print(f"Request Volume- RMSE: {req_vol_rmse}")
        #
        #
        # resp_time_rmse = mean_squared_error(test['resp_time'], pred_resp_time, squared=False)
        # print(f"Response time- RMSE: {resp_time_rmse}")
        #
        #
        # err_cnt_rmse = mean_squared_error(test['err_cnt'], pred_err_cnt, squared=False)
        # print(f"Error count- RMSE: {err_cnt_rmse}")
    print_metrics(sid,rmses,'RMSE')
    print_metrics(sid,maes,'MAE')
    print_metrics(sid,mapes,'MAPE')
    print_metrics(sid,mases,'MASE')

    end_time = timer()
    training_time = end_time - start_time
    print(f"Reraining completed for {sid}. Training time: {training_time} seconds.")


    return updated_model_req_vol,updated_model_resp_time,updated_model_err_cnt

# def predict_data(sid,app_id,start_date,end_date,freq,filtered_df):
#     model_req_vol,model_resp_time,model_err_cnt=prediction_input(sid)
#     future_w_features=predict_future_dataset(app_id,model_req_vol,model_resp_time,model_err_cnt,sid,freq,start_date,end_date,filtered_df)
#     # freq='1H'
#     return future_w_features


def forecast_future(app_id,start_date,end_date,freq,filtered_df,sid=None):
    model_req_vol,model_resp_time,model_err_cnt=prediction_input(sid)
    if model_req_vol and model_resp_time and model_err_cnt:
        actual_future_dataset = predict_future_dataset(app_id, model_req_vol, model_resp_time, model_err_cnt, sid, freq, start_date, end_date, filtered_df)
        csv_file_path = os.path.join(project_path, predictions_path,f'{sid}_predicted.csv')  # Provide the desired file path
        actual_future_dataset.to_csv(csv_file_path, index=False)
        print(f"DataFrame saved to: {csv_file_path}")
    else:
        print("Error: Unable to proceed with prediction. Check the availability of model files.")
    return actual_future_dataset
def retrain_model(model_req_vol, model_err_cnt, model_resp_time, df,sid):
    # Assuming df is the new data for retraining

    # Convert 'datetime' column to datetime object
    # df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    # df = df.set_index('datetime')
    # df.index = pd.to_datetime(df.index, format="%b %d, %Y @ %H:%M:%S.%f")

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
    for train_idx, val_idx in tss.split(df):

        req_vol_preds = []
        resp_time_preds = []
        err_cnt_preds=[]

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
        updated_model_req_vol = xgb.train({'objective': 'reg:squarederror'}, dnew_req_vol, xgb_model=model_req_vol, num_boost_round=10)


        new_X_resp_time =test[FEATURES]
        new_y_resp_time = test[RESP_TIME_TARGET]
        dnew_resp_time = xgb.DMatrix(new_X_resp_time, label=new_y_resp_time)
        updated_model_resp_time = xgb.train({'objective': 'reg:squarederror'}, dnew_resp_time, xgb_model=model_resp_time, num_boost_round=10)


        new_X_err_cnt = test[FEATURES]
        new_y_err_cnt = test[ERR_CNT_TARGET]
        dnew_err_cnt = xgb.DMatrix(new_X_err_cnt, label=new_y_err_cnt)
        updated_model_err_cnt = xgb.train({'objective': 'reg:squarederror'}, dnew_err_cnt, xgb_model=model_err_cnt, num_boost_round=10)


        # # Convert the new data into DMatrix format
        # dnew_req_vol = xgb.DMatrix(new_X_req_vol, label=new_y_req_vol)
        # dnew_resp_time = xgb.DMatrix(new_X_resp_time, label=new_y_resp_time)
        # dnew_err_cnt = xgb.DMatrix(new_X_err_cnt, label=new_y_err_cnt)
        #
        # param_req_vol = {
        #     'objective': 'reg:squarederror',
        #     'base_score':0.5, 'booster':'gbtree',
        #     'n_estimators':800,
        #     'max_depth':15,
        #     'learning_rate':0.01
        # }
        # param_resp_time = {
        #     'objective': 'reg:squarederror',
        #     'base_score':0.5, 'booster':'gbtree',
        #     'n_estimators':800,
        #     'max_depth':15,
        #     'learning_rate':0.01
        # }
        # param_err_cnt = {
        #     'objective': 'reg:squarederror',
        #     'base_score':0.5, 'booster':'gbtree',
        #     'n_estimators':800,
        #     'max_depth':15,
        #     'learning_rate':0.2
        # }
        #
        #
        #
        # updated_model_req_vol = xgb.train(param_req_vol, dnew_req_vol, xgb_model=model_req_vol)
        # updated_model_resp_time = xgb.train(param_resp_time, dnew_resp_time, xgb_model=model_resp_time)
        # updated_model_err_cnt = xgb.train(param_err_cnt, dnew_err_cnt, xgb_model=model_err_cnt)

        # Print updated model parameters
        print("Updated Model Parameters:")
        print(updated_model_req_vol.get_dump()[0])

        # Make predictions with the updated model

        pred_req_vol = updated_model_req_vol.predict(dnew_req_vol)
        pred_resp_time = updated_model_resp_time.predict(dnew_resp_time)
        pred_err_cnt = updated_model_err_cnt.predict(dnew_err_cnt)



        # test = test[:len(pred_req_vol)]
        req_vol_rmse,resp_time_rmse,err_cnt_rmse,req_vol_mae,resp_time_mae,err_cnt_mae,req_vol_mape,resp_time_mape,err_cnt_mape,req_vol_mase,resp_time_mase,err_cnt_mase=evaluate_metrics(new_y_req_vol, pred_req_vol,new_y_resp_time, pred_resp_time,new_y_err_cnt, pred_err_cnt)

        rmses.append((req_vol_rmse, resp_time_rmse,err_cnt_rmse))
        maes.append((req_vol_mae,resp_time_mae,err_cnt_mae))
        mapes.append((req_vol_mape,resp_time_mape,err_cnt_mape))
        mases.append((req_vol_mase,resp_time_mase,err_cnt_mase))

    # # You can further evaluate the performance on a test set or do additional checks
    # req_vol_rmse = mean_squared_error(test['req_vol'], pred_req_vol, squared=False)
    # print(f"Request Volume- RMSE: {req_vol_rmse}")
    #
    #
    # resp_time_rmse = mean_squared_error(test['resp_time'], pred_resp_time, squared=False)
    # print(f"Response time- RMSE: {resp_time_rmse}")
    #
    #
    # err_cnt_rmse = mean_squared_error(test['err_cnt'], pred_err_cnt, squared=False)
    # print(f"Error count- RMSE: {err_cnt_rmse}")
    print_metrics(sid,rmses,'RMSE')
    print_metrics(sid,maes,'MAE')
    print_metrics(sid,mapes,'MAPE')
    print_metrics(sid,mases,'MASE')


    # Save the updated model
    # service_name = service_name.replace('/', '_')
    updated_model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
    updated_model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
    updated_model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))


    print(f"Retraining completed for service {sid}")
    return updated_model_req_vol,updated_model_resp_time,updated_model_err_cnt
def main(mode='retrain'):


    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')
    app_id="753"


    # for service_name in unique_services:
    if mode == 'train':
        start_time=1699900200000
        end_time= 1701541800000
        df=connectelk_retrive_data(app_id,start_time,end_time)
        print(df.head())

        # data_eda(df)
        df_unique_services=update_unique_services(df)

        # vec_data, venc_data, max_correlations = classify_services(df)

        # bkt_df = list_unique_services(vec_data, max_correlations)
        # Sort the DataFrame by 'sid' in ascending order
        df_unique_services= df_unique_services.sort_values(by='sid')

        csv_file_path =os.path.join(project_path, training_path, 'all_services.csv')  # Provide the desired file path
        df_unique_services.to_csv(csv_file_path, index=False)
        print(f"DataFrame saved to: {csv_file_path}")


        # # Train models for request count, error count, and response time
        # model_req_vol, model_err_cnt, model_resp_time = train_and_forecast_service(df)
        #
        # # Optionally, you can save the trained models
        # model_req_vol.save_model(os.path.join(project_path, models_path, 'model_req_vol.json'))
        # model_err_cnt.save_model(os.path.join(project_path, models_path, 'model_err_cnt.json'))
        # model_resp_time.save_model(os.path.join(project_path, models_path, 'model_resp_time.json'))
        # if model_req_vol is not None:
        #
        #     pred_start_date='2023-12-11 11:00:00'
        #     pred_end_date='2023-12-12 11:00:00'
        #     freq='5T'
        #     forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,df)
        #     # save_forecast_index(forecast)


        services=df_unique_services['sid']
        services=[1021]
        for sid in services[:1]:
            # Train the model
            filtered_df = df[df['sid'] == sid]
            print(filtered_df.head())
            filtered_df= filtered_df.sort_values(by='record_time')

            csv_file_path = os.path.join(project_path, training_path,f'{sid}_train.csv')  # Provide the desired file path
            filtered_df.to_csv(csv_file_path, index=False)
            print(f"DataFrame saved to: {csv_file_path}")

            # train_and_forecast_service(filtered_df, sid)
            model_req_vol, model_resp_time, model_err_cnt = train_and_forecast_service(filtered_df, sid)
            # service_name = service_name.replace('/', '_')
            model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
            model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
            model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))

            if model_req_vol is not None:

                pred_start_date='2023-12-11 11:00:00'
                pred_end_date='2023-12-22 11:00:00'
                freq='5T'
                forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
                # save_forecast_index(forecast)

    elif mode == 'retrain':

            # date=datetime.now() - timedelta(days=1)
            # start_time = date.timestamp() * 1000  # Convert to milliseconds
            # end_time = (date + timedelta(days=1)).timestamp() * 1000
            start_time= 1700179200000
            end_time=1701801000000
            print(start_time)
            print(end_time)
            df = connectelk_retrive_data(app_id, start_time, end_time)

            if df is not None:
                print(df.head())

                daily_unq_services=update_unique_services(df)
                daily_unq_services= daily_unq_services.sort_values(by='sid')
                csv_file_path = os.path.join(project_path,training_path,'daily_services.csv') # Provide the desired file path
                daily_unq_services.to_csv(csv_file_path, index=False)
                print(f"DataFrame saved to: {csv_file_path}")

                # daily_services=daily_unq_services['sid']
                daily_services=[1021]
                for sid in daily_services[:1]:
                # RETrain the model
                    filtered_df = df[df['sid'] == sid]
                    print(filtered_df.head())
                    filtered_df= filtered_df.sort_values(by='record_time')

                    csv_file_path = os.path.join(project_path, retraining_path,f'{sid}_retrain.csv')  # Provide the desired file path
                    filtered_df.to_csv(csv_file_path, index=False)
                    print(f"DataFrame saved to: {csv_file_path}")

                    if not filtered_df.empty:
                        print(filtered_df.head())
                        # service_name = service_name.replace('/', '_')
                        # check the model exists
                        # model_req_vol = os.path.join(project_path, models_path,f'{sid}_req_vol.json')  # Update with the actual file names
                        # model_resp_time = os.path.join(project_path, models_path,f'{sid}_resp_time.json')
                        # model_err_cnt = os.path.join(project_path, models_path,f'{sid}_err_cnt.json')
                        # if os.path.exists(model_req_vol) and os.path.exists(model_err_cnt) and os.path.exists(model_resp_time):
                        #     model_req_vol, model_resp_time, model_err_cnt=retrainn_model(model_req_vol, model_err_cnt, model_resp_time, filtered_df,sid)
                        #     # Save the updated model
                        #     # service_name = service_name.replace('/', '_')
                        #     model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
                        #     model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
                        #     model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))

                        pred_start_date='2023-12-11 11:00:00'
                        pred_end_date='2023-12-22 11:00:00'
                        # pred_start_date='2023-11-22 11:00:00'
                        # pred_end_date='2023-12-11 11:00:00'
                        freq='5T'
                        forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
                        save_forecast_index(forecast)

                        # else:
                        #     print(f"Model files not found for {sid}. Train new model first.")
                        #     model_req_vol, model_resp_time, model_err_cnt = train_and_forecast_service(filtered_df, sid)
                        #     model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
                        #     model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
                        #     model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))
                        #     if model_req_vol is not None:
                        #
                        #         pred_start_date='2023-12-11 11:00:00'
                        #         pred_end_date='2023-12-22 11:00:00'
                        #         freq='5T'
                        #         forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
                        #         save_forecast_index(forecast)

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
    else:
                print("Invalid mode. Please enter 'train' or 'retrain'.")






        # plot_predictions(filtered_df,future_dataset,sid,freq)


#
    # # # Schedule the task to run every end of day
    # # schedule.every().day.at("15:30").do(main)
    # #
    # # # Keep the script running to allow scheduled tasks to execute
    # # while True:
    # #     schedule.run_pending()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or retrain the model.')
    parser.add_argument('mode', choices=['train', 'retrain'], default='train', help='Specify train or retrain mode')

    args = parser.parse_args()
    main(args.mode)



