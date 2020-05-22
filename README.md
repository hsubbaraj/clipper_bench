For each model, deploy clipper on a provisioned node:

1. git clone https://github.com/ucbrise/clipper.git
2. pip3 install -e clipper/clipper_admin/

modify https://github.com/ucbrise/clipper/blob/develop/clipper_admin/clipper_admin/docker/docker_container_manager.py#L395
to add "OMP_NUM_THREADS": 4 as dict entry

start clipper ie utils/start_clipper.py
note: extra_container_kwargs={'cpuset_cpus':'0-35'}) 0-35 available cpus on that node



write predict function (ie incept.py)


write client (ie parallel_bench.py)
the format for the REST endpoint is http://{publicIP}:1337/{app-name}/predict


