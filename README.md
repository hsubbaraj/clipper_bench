For each model, deploy clipper on a provisioned node:

1. git clone https://github.com/ucbrise/clipper.git
2. pip3 install -e clipper/clipper_admin/

3. modify https://github.com/ucbrise/clipper/blob/develop/clipper_admin/clipper_admin/docker/docker_container_manager.py#L395
to add "OMP_NUM_THREADS": 4 as dict entry

4. start clipper ie https://github.com/hsubbaraj/clipper_bench/blob/master/utils/start_clipper.py
note: extra_container_kwargs={'cpuset_cpus':'0-35'}) 0-35 available cpus on that node



5. write predict function (ie https://github.com/hsubbaraj/clipper_bench/blob/master/incept_setup.py)


6. write client (ie https://github.com/hsubbaraj/clipper_bench/blob/master/parallel_bench.py)
the format for the REST endpoint is http://{publicIP}:1337/{app-name}/predict


