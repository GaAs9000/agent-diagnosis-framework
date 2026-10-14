from ..core.registry import Registry
from .evaluator import DockerEvaluator
from .lhtb import LHTBEvaluator
from .swebench_pro import SWEBenchProEvaluator

def registry():
    r = Registry('l1.evaluators')
    r.register('docker-verifier', DockerEvaluator)
    r.register('lhtb-verifier', LHTBEvaluator)
    r.register('swebench_pro-verifier', SWEBenchProEvaluator)
    return r
