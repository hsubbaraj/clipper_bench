import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import io
import base64
import pyarrow as pa
import requests
import json
import time
from clipper_admin import ClipperConnection, DockerContainerManager
import clipper_admin.deployers.pytorch as pytorch_deployer

resnet18 = torchvision.models.resnet18(pretrained=True)

class Classifier:
    def __init__(self):
        self.model = None
        self.transform_pipeline  = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    def _load_model(self):
        assert torch.cuda.is_available()
        from torch_yolo.models import Darknet
        self.confidence = 0.8
        self.nms = 0.4

        self.model = Darknet(
            '/opt/conda/lib/python3.7/site-packages/torch_yolo/config/yolov3.cfg',
            img_size=224)
        self.model.load_darknet_weights('/yolo-v3.weights')

        with open('/yolo-v3.classes', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.model.cuda()

    def predict(self, model, inputs):
        import torch
        from torch_yolo.utils.utils import non_max_suppression
        if self.model == None:
                self._load_model()

        x = [pa.deserialize(i) for i in inputs]
        #print(type(x))
        #print(len(x))
        #print(type(x[0]))
        #print(type(x[0][0]))
        frame_list = x[0]
        #print("f_list len: ", len(frame_list))
        #print("f_list type: ", type(frame_list[0]))
        #t1 = self.transform_pipeline(frame_list[0].astype(np.uint8)).unsqueeze(0)
        #print(t1.shape)
        #input_tensor = torch.cat([self.transform_pipeline(frame_list[i].astype(np.uint8)).unsqueeze(0) for i in range(len(frame_list))])
        #print(frame_list[0][0].shape)
        #input_tensor = torch.cat([torch.from_numpy(frame_list[i]).unsqueeze(0) for i in range(len(frame_list))])
        #input_tensor = input_tensor.cuda()

        input_list = [torch.from_numpy(img) for img in frame_list]
        input_tensor = torch.stack(input_list, dim=0).cuda()
        with torch.no_grad():
            detections = self.model(input_tensor)
            detections = non_max_suppression(detections, self.confidence, self.nms)
        result = []
        for idx in range(len(inputs)):
            if detections[idx] != None:
                cls = self.classes[int(detections[idx][torch.argmax(detections[idx],
                                                                    dim=0)[5]][6])]
            else:
                cls = 'NONE'
            result.append(cls)
        return result




def setup_clipper():
  app_name = 'classify-app'
  model_name = 'classify-model'
  clipper_conn = ClipperConnection(DockerContainerManager(gpu=True))
  clipper_conn.connect()

  pytorch_deployer.deploy_pytorch_model(clipper_conn=clipper_conn,
          name=model_name,
          version='1',
          input_type='bytes',
          func=Classifier().predict,
          pytorch_model=resnet18,
          base_image='hsubbaraj/yolo:latest',
          batch_size=1,
          num_replicas=1,
          pkgs_to_install=['pillow', 'pyarrow', 'torch', 'torchvision'])

  clipper_conn.register_application(name=app_name,
          input_type="bytes",
          default_output="-1.0",
          slo_micros=10000000)  # 10s

  clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)

setup_clipper()
