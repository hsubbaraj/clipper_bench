from clipper_admin import ClipperConnection, DockerContainerManager

clipper_conn = ClipperConnection(DockerContainerManager(extra_container_kwargs={'cpuset_cpus':'0-35'}))
clipper_conn.start_clipper(cache_size=0)