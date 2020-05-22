import numpy as np
from clipper_admin import ClipperConnection, DockerContainerManager
import clipper_admin.deployers.pytorch as pytorch_deployer
from clipper_admin.deployers import python as python_deployer 
from clipper_admin.deployers.python import deploy_python_closure
import pickle



def join_predict(inputs):
  def _predict_one(one_input):
    prediction_matrix = pickle.loads(one_input)
    print(prediction_matrix)
    return prediction_matrix[np.argmax(prediction_matrix, dim=0)[0]][1]
  return [str(_predict_one(i)) for i in inputs]


def setup_clipper():

  app_name = 'predict-app'
  model_name = "predict-model"
  clipper_conn = ClipperConnection(DockerContainerManager())
  clipper_conn.connect()

  deploy_python_closure(
    clipper_conn,
    name="predict-model",
    version='1',
    input_type="bytes",
    func=join_predict)

  clipper_conn.register_application(name=app_name,
          input_type="bytes",
          default_output="-1.0",
          slo_micros=10000000)  # 10s

  clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)

  print("url: ", "http://{addr}/{app_name}/predict".format(addr="",app_name=app_name))
  
setup_clipper()


