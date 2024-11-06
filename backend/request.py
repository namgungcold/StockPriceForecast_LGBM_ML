import json
import gradio as gr
import requests

def request_price_prediction(data_list):
    endpoint=""
    headers={
        
    }
    body={
    }

    response=requests.post(endpoint,headers=headers,json=body)

    if response.status_code == 200:
        response_json=response.json()
        return response_json["Results"]["WebServiceOutput0"]
    else:
        return list()