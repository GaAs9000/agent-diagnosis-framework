"""Actionable preflight; never contacts the paid model endpoint."""
from __future__ import annotations
import os
import platform
from pathlib import Path
from .orchestrator import components


def doctor(config):
    checks=[]; comp=None
    def check(name,action):
        try:
            result=action(); checks.append({'name':name,'passed':True,'detail':result})
            return result
        except Exception as e:
            checks.append({'name':name,'passed':False,'error_type':type(e).__name__,'detail':str(e)})
            return None
    comp=check('component_registration',lambda:components(config))
    # Component instances are not serializable and do not belong in receipts.
    if checks and checks[0]['passed']: checks[0]['detail']='all selected factories loaded without legacy/Analyst imports'
    if comp:
        check('native_task_and_assets',lambda:{'task_id':comp['benchmark'].load(config['benchmark']['task_id']).task_id})
        check('agent_docker',comp['runtime'].preflight)
        def grade_image():
            from ..runtime.docker import docker_json
            x=docker_json('image','inspect',config['evaluation']['image'])[0]
            return {'image_id':x['Id'],'os':x.get('Os')}
        check('grader_image',grade_image)
    def credential():
        name=config['model']['api_key_env']
        if name and not os.environ.get(name): raise ValueError(f'Set environment variable {name}; do not put secrets in config')
        return {'credential_env':name,'secret_recorded':False,'model_contacted':False}
    check('credential_presence',credential)
    return {'python':platform.python_version(),'platform':platform.system(),
            'ready':all(x['passed'] for x in checks),'checks':checks,'model_endpoint_contacted':False,
            'note':'A green preflight is not a model tool-calling qualification or official scoring acceptance'}
