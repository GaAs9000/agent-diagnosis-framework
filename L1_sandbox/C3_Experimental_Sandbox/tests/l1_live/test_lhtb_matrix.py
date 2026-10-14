from __future__ import annotations
import shutil
from pathlib import Path
import pytest
from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.runtime.docker import DockerRuntime
from c3_sandbox.l1.evidence.live import verify
from c3_sandbox.l1.benchmarks.registry import registry as benchmark_registry
from c3_sandbox.l1.evaluation.registry import registry as eval_registry
from .conftest import ScriptedGemini, gemini_call, gemini_done, ROOT


def test_lhtb_task_load_and_provenance():
    reg = benchmark_registry()
    adapter = reg.create('lhtb-native', {'data_dir': str(ROOT / 'benchmarks/lhtb/data')})
    tasks = adapter.list_tasks()
    assert any(t['task_id'] == 'sokoban' for t in tasks)

    task = adapter.load('sokoban')
    assert task.benchmark == 'lhtb'
    assert task.task_id == 'sokoban'
    assert task.provenance['adapter'] == 'lhtb-native'
    assert task.provenance['data_lock_status'] == 'sha256_locked'
    assert task.evaluation.kind == 'lhtb-verifier'
    assert 'source/tests/test_outputs.py' in task.private_files
    assert any(a.name == 'bin/sokoban' for a in task.assets)


def test_lhtb_evaluator_scoring_matrix(tmp_path):
    bench_reg = benchmark_registry()
    eval_reg = eval_registry()
    adapter = bench_reg.create('lhtb-native', {'data_dir': str(ROOT / 'benchmarks/lhtb/data')})
    task = adapter.load('sokoban')
    evaluator = eval_reg.create('lhtb-verifier', {'image': 'zli12321/lhtb-sokoban:20260615'})

    # 1. Negative case: empty moves log
    sub_neg = tmp_path / 'sub_neg'; sub_neg.mkdir()
    out_neg = tmp_path / 'out_neg'
    (sub_neg / 'moves.log').write_text('')
    grade_neg = evaluator.evaluate(task.evaluation, sub_neg, out_neg)
    assert grade_neg['projection']['evaluation_validity'] == 'valid'
    assert grade_neg['projection']['task_outcome'] == 'failure'
    assert grade_neg['official_raw']['band'] == 0

    # 2. Positive case: Level 1 winning moves
    sub_pos = tmp_path / 'sub_pos'; sub_pos.mkdir()
    out_pos = tmp_path / 'out_pos'
    seq = 'DLURRRDLULLDDRULURUULDRDDRR' + 'ULDLUU'
    (sub_pos / 'moves.log').write_text('\n'.join(seq) + '\n')
    grade_pos = evaluator.evaluate(task.evaluation, sub_pos, out_pos)
    assert grade_pos['projection']['evaluation_validity'] == 'valid'
    assert grade_pos['projection']['task_outcome'] == 'success'
    assert grade_pos['official_raw']['band'] == 1
    assert grade_pos['official_raw']['reward'] > 0


def test_lhtb_end_to_end_docker_execution(tmp_path):
    out = (tmp_path / 'episode').resolve()
    cfg = resolve(ROOT / 'configs/l1/lhtb_sokoban.gemini-35lite.json')

    cmd1 = 'bash /workspace/input/bin/sokoban move DLURRRDLULLDDRULURUULDRDDRR'
    cmd2 = 'bash /workspace/input/bin/sokoban move ULDLUU'
    model = ScriptedGemini(cfg['model'], [
        gemini_call('run', {'command': cmd1}),
        gemini_call('run', {'command': cmd2}),
        gemini_done()
    ])
    runtime = DockerRuntime(cfg['runtime'])
    eval_reg = eval_registry()
    evaluator = eval_reg.create('lhtb-verifier', cfg['evaluation'])
    bench_reg = benchmark_registry()
    benchmark = bench_reg.create('lhtb-native', cfg['benchmark'])

    res = run(cfg, out, injected={
        'model': model,
        'runtime': runtime,
        'evaluation': evaluator,
        'benchmark': benchmark
    })

    assert res['status']['task_outcome'] == 'success'
    assert res['status']['evaluation_validity'] == 'valid'
    v = verify(out)
    assert v['verified'], f"LHTB evidence verification errors: {v['errors']}"
    assert not v['errors']
