import torch
import torchvision
import numpy as np
import os
import io
import base64
import requests
import json
import time
import pyarrow as pa
from PIL import Image
from torchvision import transforms
from clipper_admin import ClipperConnection, DockerContainerManager
import clipper_admin.deployers.pytorch as pytorch_deployer
from clipper_admin.deployers import python as python_deployer
from clipper_admin.deployers.python import deploy_python_closure


incept = torchvision.models.inception_v3(pretrained=True)
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def incept_predict(model, inputs):
  model.eval()
  x = [pa.deserialize(i) for i in inputs]
  input_tensor = torch.cat([transform_pipeline(i[0]).unsqueeze(0) for i in x])
  
  
  with torch.no_grad():
    out = model(input_tensor)
  
  _, indices = torch.sort(out, descending=True)
  percentage = torch.nn.functional.softmax(out, dim=1)
  p_2 = percentage.detach().numpy()
  return [[indices.data.numpy()[idx][0].item(), p_2[idx][indices[idx][0]].item()*100] for idx in range(len(inputs))]



def setup_clipper():
  app_name = 'inceptionv3-app'
  model_name = 'inceptionv3-model'
  clipper_conn = ClipperConnection(DockerContainerManager())
  clipper_conn.connect()

  pytorch_deployer.deploy_pytorch_model(clipper_conn=clipper_conn,
          name=model_name,
          version='1',
          input_type='bytes',
          func=incept_predict,
          pytorch_model=incept,
          num_replicas=10,
          batch_size = 1,
          pkgs_to_install=['pillow','pyarrow', 'torch', 'torchvision'])

  clipper_conn.register_application(name=app_name,
          input_type="bytes",
          default_output="-1.0",
          slo_micros=10000000)  # 10s

  clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)


  print("url: ", "http://{addr}/{app_name}/predict".format(addr="",app_name=app_name))


setup_clipper()