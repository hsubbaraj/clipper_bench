import numpy as np
from PIL import Image
import os
import io
import base64
import requests
import json
import time
from utils.latency_stats import print_latency_stats
import multiprocessing

def benchmark():
  resnet_url = "http://54.87.17.51:1337/resnet101-app/predict"
  inception_url = "http://34.226.193.32:1337/inceptionv3-app/predict"
  predict_url = "http://34.232.48.232:1337/predict-app/predict"

  headers = {"Content-type": "application/json"}
  latencies = []
  resnet_latencies = []
  incept_latencies = []
  pred_latencies = []
  for filename in os.listdir("imagenet_sample/imagenet/"):
    start = time.time()
    #print(filename)
    # Creating image input
    req_json = json.dumps({ "input": base64.b64encode(open("imagenet_sample/imagenet/"+filename, "rb").read()).decode() })

    #Calling resnet
    r = requests.post(resnet_url, headers=headers, data=req_json)
    resnet_output = r.json()['output']
    resnet_end = time.time()
    #Calling inception
    if resnet_output[1] < 85:
      r2 = requests.post(inception_url, headers=headers, data=req_json)
      inception_output = r2.json()['output']
    else:
      inception_output = [0, 0]
    incept_end = time.time()
    #calling predict
    predict_input = [resnet_output[1], inception_output[1], float(resnet_output[0]), float(inception_output[1])]

    req_json = json.dumps({ "input": predict_input })
    r3 = requests.post(predict_url, headers=headers, data=req_json)

    end = time.time()
    latency = (end - start)
    #print("'%s', %f ms" % (r.text, latency))
    latencies.append(latency)
    resnet_latencies.append(resnet_end-start)
    incept_latencies.append(incept_end-resnet_end)
    pred_latencies.append(end-incept_end)

  return latencies, resnet_latencies, incept_latencies, pred_latencies


print("Starting benchmark")
l1, l2, l3, l4= benchmark()
print_latency_stats(l1, "E2E")
print_latency_stats(l2, "Resnet")
print_latency_stats(l3, "Incept")
print_latency_stats(l4, "predict")