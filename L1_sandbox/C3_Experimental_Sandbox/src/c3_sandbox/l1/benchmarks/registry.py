from ..core.registry import Registry
from .officeval.adapter import NativeOfficeVal, BundleOfficeVal
from .lhtb.adapter import NativeLHTB
from .swebench_pro.adapter import NativeSWEBenchPro

def registry():
    r = Registry('l1.benchmarks')
    r.register('officeval-native', NativeOfficeVal)
    r.register('officeval-bundle', BundleOfficeVal)
    r.register('lhtb-native', NativeLHTB)
    r.register('swebench_pro-native', NativeSWEBenchPro)
    return r
