"""No SDK retries, no auth in evidence, bounded body capture, native HTTP receipts."""
from __future__ import annotations
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from ..core.contracts import ConfigurationError, HTTPResult
from ..storage import utc_now

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward credentials or a conversation to another origin.
        return None

class NativeHTTP:
    def __init__(self, config: dict):
        self.config=config
        url=urllib.parse.urlsplit(config['base_url'])
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ConfigurationError('base_url must contain no credentials, query, or fragment')
        loopback=url.hostname in ('127.0.0.1','localhost','::1')
        if url.scheme!='https' and not (url.scheme=='http' and loopback and config['kind']=='local'):
            raise ConfigurationError('Use HTTPS; HTTP is allowed only for loopback local model servers')
        self.base=config['base_url'].rstrip('/')
    def post(self, url: str, body: bytes, timeout: float, *, gemini: bool=False, extra_headers: dict[str, str] | None=None) -> HTTPResult:
        headers={'Content-Type':'application/json','Accept':'application/json','User-Agent':'l1-evidence/0.2.0'}
        # Caller-controlled headers must never replace authentication, framing, or
        # the client identity. Only explicitly safe extension headers are accepted.
        for name, value in (extra_headers or {}).items():
            if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9-]+', name)
                    or name.lower() not in {'x-opencode-session'}
                    or not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value)):
                raise ConfigurationError('Invalid or unsupported extra HTTP header')
            headers[name] = value
        key_env=self.config.get('api_key_env')
        if key_env:
            value=os.environ.get(key_env)
            if not value: raise ConfigurationError(f'Missing credential environment variable {key_env}')
            if '\n' in value or '\r' in value: raise ConfigurationError('Invalid credential value')
            headers['x-goog-api-key' if gemini else 'Authorization']=value if gemini else 'Bearer '+value
        start=time.monotonic(); started=utc_now()
        data=bytearray(); status=None; err=None; limited=False; hdrs={}
        req=urllib.request.Request(url, data=body, headers=headers, method='POST')
        try:
            try:
                resp=urllib.request.build_opener(NoRedirect()).open(req,timeout=timeout)
            except urllib.error.HTTPError as e:
                resp=e
            with resp:
                status=resp.code
                for key in ('content-type','x-request-id','request-id','retry-after'):
                    if resp.headers.get(key): hdrs[key]=resp.headers[key]
                maximum=self.config['max_response_bytes']
                while True:
                    if time.monotonic()-start>=timeout: raise TimeoutError('HTTP response deadline exceeded')
                    block=resp.read1(min(65536,maximum+1-len(data)))
                    if not block: break
                    data.extend(block)
                    if len(data)>maximum:
                        del data[maximum:]; limited=True; break
        except (OSError,TimeoutError,ValueError) as exc:
            # Exception class only: urllib text may echo URLs or authentication values.
            err={'type':type(exc).__name__, 'code':'transport_error'}
        return HTTPResult(status,bytes(data),{
            'started_at':started,'finished_at':utc_now(),'duration_seconds':time.monotonic()-start,
            'status_code':status, 'headers':hdrs, 'transport_error':err,
            'body_truncated':limited, 'attempts':1, 'auth_recorded':False,
            **({'opencode_session_id': next(value for name, value in headers.items()
                                           if name.lower() == 'x-opencode-session')}
               if any(name.lower() == 'x-opencode-session' for name in headers) else {})})
