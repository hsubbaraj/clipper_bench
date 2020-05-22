import numpy as np
from PIL import Image
import os
import io
import base64
import requests
import json
import time
from utils.latency_stats import print_latency_stats

def benchmark():
  resnet_url = "http://54.87.17.51:1337/resnet101-app/predict"
  inception_url = "http://34.226.193.32:1337/inceptionv3-app/predict"
  prediction_url = "http://334.232.48.232:1337/predict-app/predict"

  headers = {"Content-type": "application/json"}
  latencies = []
  
  for filename in os.listdir("imagenet_sample/imagenet/"):
    start = time.time()
    print(filename)
    # Creating image input
    req_json = json.dumps({ "input": base64.b64encode(open("imagenet_sample/imagenet/"+filename, "rb").read()).decode() })
    
    #Calling resnet
    r = requests.post(resnet_url, headers=headers, data=req_json)
    resnet_output = r.json()['output']
    
    #Calling inception
    if resnet_output[2] > 85:
      r2 = requests.post(inception_url, headers=headers, data=req_json)
      inception_output = r2.json()['output']
    else:
      inception_output = [0, 0, 0]
    
    #calling predict
    predict_input = [resnet_output[2], inception_output[2], resnet_output[1], inception_output[1]]
    req_json = json.dumps({ "input": predict_input })
    r3 = requests.post(predict_url, headers=headers, data=req_json)

    end = time.time()
    latency = (end - start)
    print("'%s', %f ms" % (r.text, latency))
    latencies.append(latency)
    
    break
  return latencies


print("Starting benchmark")
latencies= benchmark()
print_latency_stats(latencies, "E2E")