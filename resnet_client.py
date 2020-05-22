import pandas as pd
import numpy as np
from PIL import Image
import os
import io
import base64
import requests
import json
import time

def benchmark():
  url = "http://localhost:1337/resnet101-app/predict"
  headers = {"Content-type": "application/json"}
  latencies = []
  latencies_2 = []
  for filename in os.listdir("imagenet_sample/imagenet/")[:2]:
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

  return latencies, latencies_2


def print_latency_stats(data, ident, log=False, epoch=0):
  npdata = np.array(data)
  tput = 0

  if epoch > 0:
      tput = len(data) / epoch

  mean = np.mean(npdata)
  median = np.percentile(npdata, 50)
  p75 = np.percentile(npdata, 75)
  p95 = np.percentile(npdata, 95)
  p99 = np.percentile(npdata, 99)
  mx = np.max(npdata)

  p25 = np.percentile(npdata, 25)
  p05 = np.percentile(npdata, 5)
  p01 = np.percentile(npdata, 1)
  mn = np.min(npdata)

  output = ('%s LATENCY:\n\tsample size: %d\n' +
        '\tTHROUGHPUT: %.4f\n'
        '\tmean: %.6f, median: %.6f\n' +
        '\tmin/max: (%.6f, %.6f)\n' +
        '\tp25/p75: (%.6f, %.6f)\n' +
        '\tp5/p95: (%.6f, %.6f)\n' +
        '\tp1/p99: (%.6f, %.6f)') % (ident, len(data), tput, mean,
                                     median, mn, mx, p25, p75, p05, p95,
                                     p01, p99)

  if log:
    logging.info(output)
  else:
    print(output)

print("Starting benchmark")
latencies, l2 = benchmark()
print_latency_stats(latencies, "SM")
print_latency_stats(l2, "only request")
