"""Small native tool surface; images are real bytes, not model-authored observations."""
from __future__ import annotations
from ..core.contracts import ProtocolError

TOOLS=[
 {'name':'run','description':'Execute a shell command in an isolated Linux container. Working directory is /workspace. Each call starts a fresh container. Only /workspace persists. No network. Use input/ for original files and submission/ for final deliverables.',
  'parameters':{'type':'object','properties':{'command':{'type':'string','description':'POSIX shell command, e.g. python with a heredoc'}},'required':['command']}},
 {'name':'read_image','description':'View an existing PNG/JPEG/WebP image in /workspace. Returns actual image bytes to the model.',
  'parameters':{'type':'object','properties':{'path':{'type':'string','description':'Workspace-relative image path'}},'required':['path']}}
]

def validate_call(call,images:bool=True):
    if call.name not in ('run','read_image') or (call.name=='read_image' and not images):
        raise ProtocolError('Unknown or unavailable tool: '+call.name)
    field='command' if call.name=='run' else 'path'
    if set(call.arguments)!={field} or not isinstance(call.arguments[field],str) or not call.arguments[field] or '\x00' in call.arguments[field]:
        raise ProtocolError('Invalid native tool arguments')
    if len(call.arguments[field])>262144: raise ProtocolError('Tool argument size exceeded')
