from ..core.registry import Registry
from .docker import DockerRuntime

def registry():
    r=Registry('l1.runtimes'); r.register('docker-cold',DockerRuntime); return r
