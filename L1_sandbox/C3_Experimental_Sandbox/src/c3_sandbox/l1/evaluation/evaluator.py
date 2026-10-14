from __future__ import annotations
import os
from pathlib import Path
from ..core.contracts import CapabilityError
from ..runtime.docker import DockerRuntime
from ..storage import read_json, write_bytes, write_json, file_digest, regular_file
from .officeval import interpret

class DockerEvaluator:
    def __init__(self,config): self.config=config
    def evaluate(self,spec,submission:Path,output:Path):
        if spec.kind!='python-verifier': raise CapabilityError('evaluator requires python-verifier EvaluationSpec')
        output.mkdir(parents=True,exist_ok=False)
        runtime=DockerRuntime({'image':self.config['image'],'memory':'2g','cpus':2,'pids_limit':128,
                               'output_limit_bytes':4194304})
        provenance=runtime.preflight()
        expected_worker=file_digest(Path(__file__).with_name('worker.py'))
        actual_worker=provenance.get('python_environment',{}).get('grader_worker_sha256')
        if actual_worker!=expected_worker: raise CapabilityError('Grader image worker differs from this release; rebuild the l1-office image')
        grade_dir=output/'worker-output'; grade_dir.mkdir(mode=0o700)
        if hasattr(os,'getuid') and os.getuid()==0: os.chown(grade_dir,65532,65532)
        command=['python','-I','/opt/l1/grade_worker.py','--source','/official',
                 '--entrypoint',spec.entrypoint,'--submission','/submission','--output','/grade']
        process=runtime.container([(spec.source_root,'/official',True),(submission,'/submission',True),
                                  (grade_dir,'/grade',False)],command,self.config['timeout_seconds'],cwd='/tmp')
        write_bytes(output,'stdout.bin',process.pop('stdout')); write_bytes(output,'stderr.bin',process.pop('stderr'))
        write_json(output,'process.json',process); write_json(output,'runtime.json',provenance)
        raw=None
        raw_path=grade_dir/'official.json'
        if raw_path.exists():
            regular_file(raw_path)
            try:
                if raw_path.stat().st_size>16777216: raise ValueError('grader JSON size limit')
                raw=read_json(raw_path)
            except (ValueError,UnicodeError): raw=None
        projection=interpret(raw,spec.task_id,self.config['outcome_policy'])
        infrastructure_error=None
        if process.get('stop_reason') is not None:
            infrastructure_error=process['stop_reason']
        elif process.get('container_exit_code')!=0:
            infrastructure_error='grader_process_failed'
        elif process.get('oom_killed') or process.get('output_complete') is False:
            infrastructure_error='incomplete_verifier_execution'
        elif projection['reason']=='malformed_official_result':
            infrastructure_error='malformed_official_result'
        if infrastructure_error:
            projection.update(evaluation_validity='invalid',task_outcome='unknown',reason=infrastructure_error)
        result={'projection':projection,'official_raw':raw,'process':process,'runtime':provenance,
                'official_raw_sha256':file_digest(raw_path) if raw_path.is_file() else None}
        if infrastructure_error:
            result.update(status='infra_error',error_kind='infrastructure',error=infrastructure_error)
        return result
