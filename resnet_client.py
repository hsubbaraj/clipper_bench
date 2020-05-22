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
  url = "http://54.87.17.51:1337/resnet101-app/predict"
  headers = {"Content-type": "application/json"}
  latencies = []
  latencies_2 = []
  for filename in os.listdir("imagenet_sample/imagenet/"):
    start = time.time()
    print(filename)
    req_json = json.dumps({ "input": base64.b64encode(open("imagenet_sample/imagenet/"+filename, "rb").read()).decode() })
    start_2 = time.time()
    r = requests.post(url, headers=headers, data=req_json)
    end = time.time()
    latency = (end - start)
    print("'%s', %f ms" % (r.text, latency))
    latencies.append(latency)
    latencies_2.append(end-start_2)
    break
  return latencies, latencies_2


print("Starting benchmark")
latencies, l2 = benchmark()
print_latency_stats(latencies, "SM")
print_latency_stats(l2, "only request")