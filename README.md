For each model, deploy clipper on a provisioned node:

1. Start an instance per-model, where the number of vCPUs (or GPUs) on the instance is the number of resources per-model times the number of model replicas. 

2. Install relevant Python dependencies: 
    * `torch`, `pyarrow` `scipy` for all models 
    * `torchvision` for image+video models
    * `cv2` for video models
    * `fasttext` for NMT models

3. Install Docker: instructions [here](https://docs.docker.com/engine/install/ubuntu/).

4. Clone Clipper: https://github.com/ucbrise/clipper.git

5. `pip3 install -e clipper/clipper_admin/`

6. Modify Clipper admin to set `OMP_NUM_THREADS` argument. This should be the number of threads that _each_ model container should use (i.e., 8 for the CPUs). Modify the line [here](https://github.com/ucbrise/clipper/blob/develop/clipper_admin/clipper_admin/docker/docker_container_manager.py#L395).

7. Start Clipper using the script [here](https://github.com/hsubbaraj/clipper_bench/blob/master/utils/start_clipper.py).
Note: You have to set `extra_container_kwargs={'cpuset_cpus':'0-35'})` with the number of available CPUs (total) on that node. 
TODO: What do we need to set for the number of GPUs?

8. Write a predict function for each model (example [here](https://github.com/hsubbaraj/clipper_bench/blob/master/incept_setup.py)).
NOTE: Set `num_replicas` and `batch_size` when registering the model. Need to also include `pyarrow` in `packages_to_install` 

9. Write a benchmark client (example [here](https://github.com/hsubbaraj/clipper_bench/blob/master/parallel_bench.py)).
The format for the REST endpoint is http://{publicIP}:1337/{app-name}/predict.


