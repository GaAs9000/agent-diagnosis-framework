"""Thin adapter to the EXISTING L1 native transports; no duplicate API SDK.

Install L2 into the same virtualenv as the user's L1 0.2 source checkout. Offline
facts/rules/verification do not depend on c3_sandbox, credentials or a GPU stack.
"""
from __future__ import annotations
import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from ..io import Unsupported, L2Error, sha_file


def create(config: dict):
    try:
        from c3_sandbox.l1.models.native import GeminiBackend, OpenAICompatibleBackend
    except ImportError as exc:
        raise Unsupported('L1 0.2 native backend unavailable. Install your existing C3_Experimental_Sandbox in this virtualenv; offline rules still work.') from exc
    cls = GeminiBackend if config['adapter'] == 'gemini' else OpenAICompatibleBackend
    backend = cls(copy.deepcopy(config))
    for name in ('identity', 'initial', 'request', 'send', 'parse', 'accept'):
        if not callable(getattr(backend, name, None)): raise Unsupported('Installed L1 ModelBackend contract differs: ' + name)
    return backend


def provenance(backend):
    source = inspect.getsourcefile(type(backend))
    return {'class': type(backend).__module__ + '.' + type(backend).__qualname__,
            'implementation_sha256': sha_file(Path(source)) if source and Path(source).is_file() else None,
            'backend_identity': backend.identity(), 'contract': 'L1-native-0.2'}


def observation(value):
    return SimpleNamespace(observation=value, image=None, image_mime=None, stdout=b'', stderr=b'')


def feedback(conversation, message):
    result = copy.deepcopy(conversation)
    if 'contents' in result: result['contents'].append({'role': 'user', 'parts': [{'text': message}]})
    elif 'messages' in result: result['messages'].append({'role': 'user', 'content': message})
    else: raise Unsupported('Unknown native conversation shape')
    return result


def doctor(config):
    result = {'model_endpoint_contacted': False, 'ready': False, 'checks': []}
    try:
        backend = create(config['model'])
        result['backend'] = provenance(backend)
        result['checks'].append({'name': 'native_backend_contract', 'passed': True})
        env = config['model']['api_key_env']
        result['checks'].append({'name': 'credential_presence', 'variable': env,
                                 'passed': env is None or bool(os.environ.get(env))})
        result['ready'] = all(c['passed'] for c in result['checks'])
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['note'] = 'Readiness is not a successful model invocation, calibrated diagnosis or live experiment qualification.'
    return result
