from clipper_admin import ClipperConnection, DockerContainerManager

clipper_conn = ClipperConnection(DockerContainerManager(gpu=True))
clipper_conn.start_clipper(cache_size=0)
