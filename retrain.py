# @appp.route('/retrain')

def retrain_model(df, target_column, model_name):
    start_time = timer()
    try:
        tss = TimeSeriesSplit(n_splits=2)
        df = df.sort_index()

        rmses = []
        maes = []
        mapes = []
        mases = []

        for train_idx, val_idx in tss.split(df):

            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = create_features(train)
            test = create_features(test)

            FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']

            new_X = test[FEATURES]
            new_y = test[target_column]

            dnew = xgb.DMatrix(new_X, label=new_y)
            updated_model = xgb.XGBRegressor(objective='reg:squarederror')

            grid_search = GridSearchCV(updated_model, param_grid, scoring='neg_mean_squared_error', cv=2)
            grid_search.fit(new_X, new_y)
            updated_model = grid_search.best_estimator_

            best_params = grid_search.best_params_
            print(f"Best parameters for {model_name}: {best_params}")

            pred = updated_model.predict(new_X)
            rmse, mae, mape, mase=evaluate_metrics(y_test_req_vol, req_vol_pred)

            rmses.append((rmse,))
            maes.append((mae,))
            mapes.append((mape,))
            mases.append((mase,))


            end_time = timer()
            training_time = end_time - start_time
            print(f"Retraining completed for {model_name}. Training time: {training_time} seconds.")

        return updated_model, (rmses, maes, mapes, mases)

# def print_metrics(model_name, metrics, target_column):
#     print(f"Metrics for {model_name} on {target_column}:")
#
#     rmses, maes, mapes, mases = metrics
#
#     print(f"RMSEs: {rmses}")
#     print(f"MAEs: {maes}")
#     print(f"MAPEs: {mapes}")
#     print(f"MASEs: {mases}")

def retrain_and_predict_for_service(filtered_df, sid, model_type):
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

@appp.route('/retrain')
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
    start_time= 1701648000000 # Dec 4,2023 12 am
    end_time=1701734400000 #Dec 5,2023 12 am
    print(start_time)
    print(end_time)
    df = connectelk_retrive_data(app_id, start_time, end_time,source_index)

    if df is not None:
        print(df.head())

        logging.info(df.head())
        df_all_services,df_corr_vol_err,df_corr_vol_resp=list_services_retraining(df)
        # df_all_services=[1007,1008,1021,1034]
        # df_corr_vol_err = df_corr_vol_err['sid'].tolist()[:1]
        # df_corr_vol_resp = df_corr_vol_resp['sid'].tolist()[:1]

        all_forecasts = []
        def retrain_model_for_service(sid):
            df_service  =filter_service_by_sid(df,sid)

            if not df_service.empty:
                model_req_vol = os.path.join(project_path, models_path,f'{sid}_vol.json')  # Update with the actual file names
                model_resp_time = os.path.join(project_path, models_path,f'{sid}_resp.json')
                model_err_cnt = os.path.join(project_path, models_path,f'{sid}_err.json')
                if sid in df_corr_vol_err:
                    if os.path.exists(model_err_cnt):
                    # Train and predict for error
                        df_err=remove_err_outliers(df_service)
                        err_model = retrain_and_predict_for_service(df_err, sid, 'err')
                    else:
                        start_time = 1701282600000  # start time of the training data
                        end_time = datetime.now().timestamp() * 1000
                        df = connectelk_retrive_data(app_id, start_time, end_time,training_index)
                        df_service  =filter_service_by_sid(df,sid)
                        df_err=remove_err_outliers(df_service)
                        train_and_predict_for_service(df_err, sid, 'err')

                if sid in df_corr_vol_resp:
                    if os.path.exists(model_resp_time):

                        df_resp=remove_resp_outliers(df_service)
                        # Train and predict for response time
                        resp_model = retrain_and_predict_for_service(df_resp, sid, 'resp')
                    else:
                        start_time = 1701282600000  # start time of the training data
                        end_time = datetime.now().timestamp() * 1000
                        df = connectelk_retrive_data(app_id, start_time, end_time,training_index)
                        df_service  =filter_service_by_sid(df,sid)
                        df_resp=remove_resp_outliers(df_service)
                        train_and_predict_for_service(df_resp, sid, 'resp')

                if os.path.exists(model_req_vol):
                    # Train and predict for volume
                    df_vol=remove_vol_outliers(df_service)
                    vol_model = retrain_and_predict_for_service(df_vol, sid, 'vol')
                else:
                    train_and_predict_for_service(df_vol, sid, 'vol')

                pred_start_date='2023-12-05 12:00:00'
                pred_end_date='2023-12-06 12:00:00'
                freq='5T'
                forecast=predict_future_dataset(app_id,sid,freq,pred_start_date,pred_end_date)
                csv_file_path = os.path.join(project_path, predictions_path,f'{sid}_predicted.csv')  # Provide the desired file path
                forecast.to_csv(csv_file_path, index=False)
                print(f"Prediction DataFrame saved to: {csv_file_path}")
                # save_forecast_index(forecast,target_index)
                bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
                                         elk_password)
                all_forecasts.append(forecast)

            else:
                print(f"No records found for {sid} - retraining model")
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(retrain_model_for_service, df_all_services)

        total_execution_time = timer() - st_time

        # Concatenate all forecasts into a single DataFrame
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

        # Save the combined forecasts to a CSV file
        combined_csv_path = os.path.join(project_path, predictions_path, 'combined_forecasts.csv')
        combined_forecasts.to_csv(combined_csv_path, index=False)
        print(f"Combined Forecast CSV saved to: {combined_csv_path}")

        # Delete the old target OpenSearch index
        delete_open_search_index(target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

        # Save the combined forecasts to OpenSearch
        bulk_save_forecast_index(combined_forecasts, target_index, elasticsearch_host, elasticsearch_port, elk_username, elk_password)

    # Example response
        response_data = {'status': 'success', 'message': 'Model retrained successfully', 'total_execution_time': total_execution_time}
        logging.info(response_data)

        # Return a valid HTTP response (JSON in this case)
        return jsonify(response_data)
    else:
        logging.error("Failed to load data.")
        return None


def retrain_request_volume_model(df,sid):
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
        print(f"Skipping service due to error: {e}")


    return None
def retrain_response_time_model(df,sid):
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
        print(f"Skipping service due to error: {e}")


    return None
def retrain_error_count_model(df,sid):
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
        print(f"Skipping service due to error: {e}")


    return None
# def retrain_resp_time_model_with_grid_search(df,sid):
#     start_time = timer()
#     try:
#         tss = TimeSeriesSplit(n_splits=2)
#         df = df.sort_index()
#
#         rmses = []
#         maes=[]
#         mapes=[]
#         mases=[]
#
#         # Define the hyperparameter grid for GridSearchCV
#         param_grid = {
#             'n_estimators': [500,800,1000],
#             'learning_rate': [0.01, 0.2, 0.3],
#             'max_depth': [13,15,20],
#             # Add other hyperparameters to tune
#         }
#         for train_idx, val_idx in tss.split(df):
#
#             req_vol_preds = []
#
#             train = df.iloc[train_idx]
#             test = df.iloc[val_idx]
#
#             train = create_features(train)
#             test = create_features(test)
#
#             FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
#             # 'lag1','lag2','lag3']
#
#             RESP_TIME_TARGET = 'resp_time_sum'
#
#
# # Add lags and create features
#             # df = add_lags(df)  # Assuming you have a function to add lags
#             # df = create_features(df)
#             new_X_resp_time = test[FEATURES]
#             new_y_resp_time = test[RESP_TIME_TARGET]
#
#             dnew_resp_time = xgb.DMatrix(new_X_resp_time, label=new_y_resp_time)
#
#             updated_model_resp_time = xgb.XGBRegressor(objective='reg:squarederror',base_score=0.5, booster='gbtree')
#
#             # Create a scorer for GridSearchCV using negative mean squared error
#             # scorer = make_scorer(mean_squared_error, greater_is_better=False)
#             grid_search_resp_time = GridSearchCV(updated_model_resp_time, param_grid, scoring='neg_mean_squared_error', cv=tss)
#
#             grid_search_resp_time.fit(new_X_resp_time, new_y_resp_time)
#             updated_model_resp_time = grid_search_resp_time.best_estimator_
#
#
#             # Get the best hyperparameters
#             best_params_resp_time = grid_search_resp_time.best_params_
#             print("best parameters for resp time:", best_params_resp_time)
#
#             # Make predictions with the updated model
#             # pred_req_vol = updated_model_req_vol.predict(new_X_req_vol)
#
#             resp_time_pred, resp_time_interval = predict_resp_time( updated_model_resp_time,new_X_resp_time)
#             # Append intervals to your lists
#             # req_vol_preds.append((req_vol_pred, req_vol_interval))
#
#             resp_time_rmse, resp_time_mae, resp_time_mape, resp_time_mase=evaluate_metrics(new_y_resp_time, resp_time_pred)
#
#             rmses.append((resp_time_rmse,))
#             maes.append((resp_time_mae,))
#             mapes.append((resp_time_mape,))
#             mases.append((resp_time_mase,))
#
#         print_metrics(sid,rmses,'RMSE','Response Time')
#         print_metrics(sid,maes,'MAE','Response Time')
#         print_metrics(sid,mapes,'MAPE','Response Time')
#         print_metrics(sid,mases,'MASE','Response Time')
#
#         end_time = timer()
#         training_time = end_time - start_time
#         print(f"Retraining completed for {sid} for Response Time . Retraining time: {training_time} seconds.")
#
#         # Return the final model
#         return updated_model_resp_time
#
#     except ValueError as e:
#         # Handle the exception (e.g., print a message)
#         print(f"Skipping service due to error: {e}")
#
#
#     return None
#
# def retrain_err_cnt_model_with_grid_search(df,sid):
#     start_time = timer()
#     try:
#         tss = TimeSeriesSplit(n_splits=2)
#         df = df.sort_index()
#
#         rmses = []
#         maes=[]
#         mapes=[]
#         mases=[]
#
#         # Define the hyperparameter grid for GridSearchCV
#         param_grid = {
#             'n_estimators': [500,800,1000],
#             'learning_rate': [0.01, 0.2, 0.3],
#             'max_depth': [13,15,20],
#             # Add other hyperparameters to tune
#         }
#         for train_idx, val_idx in tss.split(df):
#
#             req_vol_preds = []
#
#             train = df.iloc[train_idx]
#             test = df.iloc[val_idx]
#
#             train = create_features(train)
#             test = create_features(test)
#
#             FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year']
#             # 'lag1','lag2','lag3']
#
#             ERR_CNT_TARGET = 'error_count'
#
#
#             # Add lags and create features
#             # df = add_lags(df)  # Assuming you have a function to add lags
#             # df = create_features(df)
#             new_X_err_cnt = test[FEATURES]
#             new_y_err_cnt = test[ERR_CNT_TARGET]
#
#             dnew_err_cnt = xgb.DMatrix(new_X_err_cnt, label=new_y_err_cnt)
#             updated_model_err_cnt = xgb.XGBRegressor(objective='reg:squarederror',base_score=0.5, booster='gbtree')
#
#             # Create a scorer for GridSearchCV using negative mean squared error
#             # scorer = make_scorer(mean_squared_error, greater_is_better=False)
#             grid_search_err_cnt = GridSearchCV(updated_model_err_cnt, param_grid, scoring='neg_mean_squared_error', cv=tss)
#             grid_search_err_cnt.fit(new_X_err_cnt, new_y_err_cnt)
#             updated_model_err_cnt = grid_search_err_cnt.best_estimator_
#
#
#             # Get the best hyperparameters
#             best_params_err_cnt = grid_search_err_cnt.best_params_
#
#             print("best parameters for resp time:", best_params_err_cnt)
#
#             # Make predictions with the updated model
#             # pred_req_vol = updated_model_err_cnt.predict(new_X_req_vol)
#
#             err_cnt_pred, err_ent_interval = predict_err_cnt( updated_model_err_cnt,new_X_err_cnt)
#             # Append intervals to your lists
#             # req_vol_preds.append((req_vol_pred, req_vol_interval))
#
#             err_cnt_rmse, err_cnt_mae, err_cnt_mape, err_cnt_mase=evaluate_metrics(new_y_err_cnt, err_cnt_pred)
#
#             rmses.append((err_cnt_rmse,))
#             maes.append((err_cnt_mae,))
#             mapes.append((err_cnt_mape,))
#             mases.append((err_cnt_mase,))
#
#         print_metrics(sid,rmses,'RMSE','Error count')
#         print_metrics(sid,maes,'MAE','Error count')
#         print_metrics(sid,mapes,'MAPE','Error count')
#         print_metrics(sid,mases,'MASE','Error count')
#
#         end_time = timer()
#         training_time = end_time - start_time
#         print(f"Retraining completed for {sid} for Error Count . Retraining time: {training_time} seconds.")
#
#         # Return the final model
#         return updated_model_err_cnt
#
#     except ValueError as e:
#         # Handle the exception (e.g., print a message)
#         print(f"Skipping service due to error: {e}")
#
#
#     return None