import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import io
import base64
import requests
import json
import time
from clipper_admin import ClipperConnection, DockerContainerManager
import clipper_admin.deployers.pytorch as pytorch_deployer



resnet101 = torchvision.models.resnet101(pretrained=True)
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

def resnet_predict(model, inputs):
  assert torch.cuda.is_available()
  model.cuda()
  model.eval()
  start = time.time()
  pil_images = [Image.open(io.BytesIO(i)).convert("RGB") for i in inputs]
  input_tensor = torch.cat([transform_pipeline(i).unsqueeze(0) for i in pil_images])
  input_tensor = input_tensor.cuda()
  end1 = time.time()
  with torch.no_grad():
    out = model(input_tensor)
  end2 = time.time()
  print("transform: ", end1-start)
  print("model: ", end2-end1)
  _, indices = torch.sort(out, descending=True)
  percentage = torch.nn.functional.softmax(out, dim=1)
  p_2 = percentage.detach().cpu().numpy()
  return [[indices.detach().cpu().numpy()[idx][0].item(), p_2[idx][indices[idx][0]].item()*100] for idx in range(len(inputs))]



def setup_clipper():
  app_name = 'resnet101-app'
  model_name = 'resnet101-model'
  clipper_conn = ClipperConnection(DockerContainerManager(gpu=True))
  clipper_conn.connect()

  pytorch_deployer.deploy_pytorch_model(clipper_conn=clipper_conn,
          name=model_name,
          version='1',
          input_type='bytes',
          func=resnet_predict,
          pytorch_model=resnet101,
          batch_size=1,
          num_replicas=1,
          pkgs_to_install=['pillow', 'torch', 'torchvision'])

  clipper_conn.register_application(name=app_name,
          input_type="bytes",
          default_output="-1.0",
          slo_micros=10000000)  # 10s

  clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)
  print("query_adress: ", clipper_conn.get_query_addr())
  print("app_name: ", )
  print("model_name: ", )
  print("url: ", "http://{addr}/{app_name}/predict".format(addr=clipper_conn.get_query_addr(),app_name=app_name))


setup_clipper()
