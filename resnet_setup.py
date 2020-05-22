import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import io
import base64
import requests
import json
import time
from clipper_admin import ClipperConnection, DockerContainerManager
import clipper_admin.deployers.pytorch as pytorch_deployer




resnet101 = torchvision.models.resnet18(pretrained=True)
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
  pil_images = [Image.open(io.BytesIO(i) for i in inputs)]
  input_tensor = torch.cat([self.preprocessor(i).unsqueeze(0) for i in pil_images])
  input_tensor = input_tensor.cuda()
  with torch.no_grad():
    output_tensor = self.model(input_tensor)

  _, indices = torch.sort(out, descending=True)
  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
  p_2 = percentage.detach().numpy()
  print("indices.shape: ", indices.shape)
  print("p2.shape: ", p2.shape)
  conf = p_2[indices[0][0]].item()
  idx = indices.data.numpy()[0][0].item()
  print("conf: ", str(conf))
  print("idx: ", idx)
  return [[0, 0] for i in inputs]

  def _predict_one(one_input_arr):
    
    model.eval()
    img = Image.open(io.BytesIO(one_input_arr))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    out = model(img)
    print("After model runs")
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    p_2 = percentage.detach().numpy()
    conf = p_2[indices[0][0]].item()
    if conf > 85:
      x = 1
    else:
      x = 0

    return [one_input_arr, indices.data.numpy()[0][0].item(),  p_2[indices[0][0]].item(), x]
  return [_predict_one(i) for i in inputs]



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
          num_replicas=1,
          batch_size=1,
          pkgs_to_install=['prometheus_client', 'zmq', 'pillow', 'torch', 'torchvision'])

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


