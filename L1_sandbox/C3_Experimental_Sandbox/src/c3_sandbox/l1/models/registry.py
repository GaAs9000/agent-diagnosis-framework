from ..core.registry import Registry
from .native import GeminiBackend, OpenAICompatibleBackend

def registry():
    r=Registry('l1.models')
    r.register('gemini',GeminiBackend)
    r.register('openai-compatible',OpenAICompatibleBackend)
    return r
