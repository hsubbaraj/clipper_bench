import numpy as np
from PIL import Image
import os
import io
import base64
import requests
import json
import pyarrow as pa
import random
import time
from utils.latency_stats import print_latency_stats
import multiprocessing

def benchmark(procnum, send_end):
  resnet_url = "http://54.87.17.51:1337/resnet101-app/predict"
  inception_url = "http://3.235.164.133:1337/inceptionv3-app/predict"
  predict_url = "http://34.232.48.232:1337/predict-app/predict"


  headers = {"Content-type": "application/json"}
  latencies = []
  post_serial_latencies = []
  resnet_latencies = []
  incept_latencies = []
  # pred_latencies = []
  y = os.listdir("imagenet_sample/imagenet/")
  x = random.choices(y,k=1000)
  count=0
  for filename in x:
    start = time.time()
    #print(filename)
    # Creating image input
    req_json = json.dumps({ "input": base64.b64encode(open("imagenet_sample/imagenet/"+filename, "rb").read()).decode() })
    serial_start = time.time()
    #Calling resnet
    r = requests.post(resnet_url, headers=headers, data=req_json)
    resnet_output = r.json()['output']
    if r.json()['default']:
        print("ERROR", os.getpid())
        return
    resnet_end = time.time()

    #Calling inception
    incept_start = time.time()
    input_bytes = pa.serialize([np.asarray(Image.open("imagenet_sample/imagenet/"+filename).convert("RGB")), resnet_output[1]]).to_buffer().to_pybytes()
    req_json = json.dumps({ "input": base64.b64encode(input_bytes).decode() })
    r2 = requests.post(inception_url, headers=headers, data=req_json)
    inception_output = r2.json()['output']
    incept_end = time.time()

    #calling predict
    predict_input = [resnet_output[1], inception_output[1], float(resnet_output[0]), float(inception_output[1])]

    req_json = json.dumps({ "input": predict_input })
    r3 = requests.post(predict_url, headers=headers, data=req_json)

    end = time.time()
    incept_end= end
    latency = (end - start)
    #print("'%s', %f ms" % (r.text, latency))
    latencies.append(latency)
    resnet_latencies.append(resnet_end-serial_start)
    incept_latencies.append(incept_end-resnet_end)
    post_serial_latencies.append(end-serial_start)
  send_end.send([latencies, post_serial_latencies,count, resnet_latencies, incept_latencies])

def main():

    proc = []
    pipe_list = []
    for i in range(10):
        recv_end, send_end = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=benchmark, args=(i, send_end))
        proc.append(p)
        pipe_list.append(recv_end)

    print(proc)

    for p in proc:
        p.start()

    for p in proc:
        p.join()

    result_list = [x.recv() for x in pipe_list]
    total_tput = 0
    all_latencies = []
    all_post_latencies = []
    all_resnet_latencies = []
    all_incept_latencies = []
    for l in range(len(result_list)):
        print("inception_count: ", result_list[l][2])
        lat = result_list[l][0]

        all_latencies.extend(lat)
        all_resnet_latencies.extend(result_list[l][3])
        all_incept_latencies.extend(result_list[l][4])
        all_post_latencies.extend(result_list[l][1])

        thread_latency = sum(lat)
        total_tput += 1000/thread_latency
        print_latency_stats(lat, "E2E Process: "+ str(l))
        #print_latency_stats(p_lat, "w/o serialization Process: "+ str(l))
    print_latency_stats(all_latencies, "E2E ALL THREADS")
    #print_latency_stats(all_post_latencies, "w/o serialization ALL THREADS")
    print_latency_stats(all_resnet_latencies, "RESNET ALL THREADS")
    print_latency_stats(all_incept_latencies, "INCEPTION ALL THREADS")
    print("Total throughput: ", total_tput)

if __name__ == '__main__':
    main()