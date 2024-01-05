def seq_train():
    # To ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    color_pal = sns.color_palette()
    plt.style.use('fivethirtyeight')

    start_time=1699900200000
    end_time= 1701541800000
    df=connectelk_retrive_data(app_id,start_time,end_time,source_index)
    print(df.head())

    df_unique_services=update_unique_services(df)

    # vec_data, venc_data, max_correlations = classify_services(df)

    # bkt_df = list_unique_services(vec_data, max_correlations)
    # Sort the DataFrame by 'sid' in ascending order
    df_unique_services= df_unique_services.sort_values(by='sid')

    csv_file_path =os.path.join(project_path, training_path, 'all_services.csv')  # Provide the desired file path
    df_unique_services.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to: {csv_file_path}")

    services=df_unique_services['sid']
    services=[1020]
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
            pred_end_date='2023-12-13 11:00:00'
            freq='1H'
            forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
            bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
                                     elk_password)
    # Use ThreadPoolExecutor for parallel processing

    # Example response
    response_data = {'status': 'success', 'message': 'Model trained successfully'}

    # Return a valid HTTP response (JSON in this case)
    return jsonify(response_data)

def train_and_predict(filtered_df,sid):
    # train_and_forecast_service(filtered_df, sid)
    model_req_vol, model_resp_time, model_err_cnt = train_and_forecast_service(filtered_df, sid)
    # service_name = service_name.replace('/', '_')
    model_req_vol.save_model(os.path.join(project_path, models_path,f'{sid}_req_vol.json'))
    model_resp_time.save_model(os.path.join(project_path, models_path,f'{sid}_resp_time.json'))
    model_err_cnt.save_model(os.path.join(project_path, models_path,f'{sid}_err_cnt.json'))

    if model_req_vol is not None:

        pred_start_date='2023-12-11 11:00:00'
        pred_end_date='2023-12-12 11:00:00'
        freq='5T'
        forecast=forecast_future(app_id,pred_start_date,pred_end_date,freq,filtered_df,sid)
        bulk_save_forecast_index(forecast, target_index, elasticsearch_host, elasticsearch_port, elk_username,
                                 elk_password)


