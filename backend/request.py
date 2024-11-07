import json
import gradio as gr
import requests

#예측 진행
def request_price_prediction(data_list):
    #ENDPOINT, METHOD, HEADER, BODY
    endpoint = "http://67688059-0432-4ea0-ba69-27cef7128005.koreacentral.azurecontainer.io/score"
    headers = {'Content-type': 'application/json',
               'Authorization': 'Bearer xaFSLMfDUGyPCO5gV6pyseAUWNY9i0Ur'
               }
    body = {
        "Inputs": {
            "input1": data_list
        }
    }
    try:
        response = requests.post(endpoint, headers=headers, json=body)
        response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx
        response_json = response.json()
        
        # Check if the expected key exists in the response
        if "Results" in response_json and "WebServiceOutput0" in response_json["Results"]:
            result_list = response_json["Results"]["WebServiceOutput0"]
            predicted_value = result_list[0]["Scored Labels"] if result_list else None
            return predicted_value
        else:
            print("Unexpected response structure:", response_json)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None
    



