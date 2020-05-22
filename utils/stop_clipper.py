from clipper_admin import ClipperConnection, KubernetesContainerManager, DockerContainerManager

clipper_conn = ClipperConnection(DockerContainerManager(gpu=True))
clipper_conn.stop_all(graceful=False)
