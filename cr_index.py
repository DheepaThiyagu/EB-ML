from elasticsearch import Elasticsearch
import configparser

import requests
from requests.auth import HTTPBasicAuth
# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
# Specify the relative path to your models directory

elasticsearch_host = config.get('elasticsearch', 'host')
elasticsearch_port = config.getint('elasticsearch', 'port')
elk_username = config.get('elasticsearch', 'username')
elk_password = config.get('elasticsearch', 'password')
target_index = config.get('indices', 'target_index')


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

# check_elk_connection(url_string)
url_string = f'{elasticsearch_host}:{elasticsearch_port}'
print(url_string)

try:
    es = Elasticsearch([url_string], basic_auth=(elk_username, elk_password),verify_certs=False)

    if es.ping():
        print("Connected to Elasticsearch")
    else:
        print("Could not connect to Elasticsearch")
except Exception as e:
    print("Error connecting to Elasticsearch:", e)

# Define the index mapping
mapping = {
    "mappings": {
        "properties": {
            "hour": {"type": "integer"},
            "dayofweek": {"type": "integer"},
            "quarter": {"type": "integer"},
            "month": {"type": "integer"},
            "year": {"type": "integer"},
            "dayofyear": {"type": "integer"},
            "dayofmonth": {"type": "integer"},
            "weekofyear": {"type": "integer"},
            "datetime": {"type": "date"},
            "app_id": {"type": "integer"},
            "sid": {"type": "integer"},
            "pred_req_vol": {"type": "integer"},
            "Req_Vol_Lower": {"type": "integer"},
            "Req_Vol_Upper": {"type": "integer"},
            "Req_Vol_Conf_score": {"type": "integer"},
            "pred_err_cnt": {"type": "integer"},
            "Err_Cnt_Lower": {"type": "integer"},
            "Err_Cnt_Upper": {"type": "integer"},
            "Err_Cnt_Conf_score": {"type": "integer"},
            "pred_resp_time": {"type": "float"},
            "Resp_Time_Lower": {"type": "float"},
            "Resp_Time_Upper": {"type": "float"},
            "Resp_Time_Conf_score": {"type": "float"},
            "Timestamp": {"type": "date"},
            "fivemin_timestamp": {"type": "date"},
            "hour_timestamp": {"type": "date"},
            "day_timestamp": {"type": "date"}
        }
    }
}

# Set the Content-Type header to application/json
headers = {"Content-Type": "application/json"}

# Create the index
index_name = target_index

try:
    # Create the index with the specified mapping
    response = es.indices.create(index=index_name, body=mapping, headers=headers)
    print("Index creation response:", response)
except Exception as e:
    print("Error creating index:", e)

