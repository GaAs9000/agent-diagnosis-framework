"""Exp1 acceptance-gap contracts with fixture ports; never invokes live services or models."""
import copy
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
import threading
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d = load('run_acceptance_gap_exp1.py', 'exp1_test_driver')
an = load('analyze_acceptance_gap_exp1.py', 'exp1_test_analysis')


@pytest.fixture
def env(tmp_path, monkeypatch):
    real, _ = d.load_spec()
    spec = copy.deepcopy(real)
    for task, entry in spec['conditions']['F']['deliverable_names'].items():
        verifier = tmp_path / f'{task}_verifier.py'
        verifier.write_text('\n'.join(repr(name) for name in entry['names']), encoding='utf-8')
        entry['verifier'] = {'path': str(verifier), 'sha256': d.file_digest(verifier)}
    base_configs = d.base.task_configs(d.base.CONFIG)
    reference = tmp_path / 'reference_freeze.json'
    d.write_json(reference, {'task_config_sha256': {t: d.digest(c) for t, c in base_configs.items()}})
    spec['pi']['reference_prospective_freeze'] = {'path': str(reference), 'sha256': d.file_digest(reference),
                                                  'use': 'fixture'}
    spec_path = tmp_path / 'spec.json'
    d.write_json(spec_path, spec)
    monkeypatch.setattr(d, 'L1', tmp_path / 'L1')
    monkeypatch.setattr(d, 'PHASE_ROOT', tmp_path / 'L2/validation/acceptance_gap_exp1_20261022')
    source = tmp_path / 'driver_source.py'
    source.write_text('fixture source')
    monkeypatch.setattr(d, 'source_files', lambda: {'driver_source.py': d.file_digest(source)})
    identity = dict(identity_receipt={'path': 'fixture', 'sha256': 'e' * 64},
                    command_after_timeout=['vllm', 'serve', 'fixture'])
    frozen = dict(actor_image_id='sha256:' + 'a' * 64, actor_user='1:1', bound_files={},
                  expected_service_identity=identity)
    monkeypatch.setattr(d, 'reference_freeze', lambda spec: copy.deepcopy(frozen))
    monkeypatch.setattr(d.base, 'validate_service_identity', lambda value, configs, bound: copy.deepcopy(value))
    review = d.preview(spec_path)['review_digest']
    auth = tmp_path / 'authorization.json'
    d.write_json(auth, dict(exact_text='开始 Exp1', source='direct user message (fixture)',
                            experiment_id=spec['experiment_id'], spec_sha256=d.file_digest(spec_path),
                            approved_review_digest=review, authorized_blocks=[0, 1]))
    now = float(int(time.time()))
    lease = tmp_path / 'lease.json'
    d.write_json(lease, dict(started_epoch=now - 1, hard_stop_epoch=now - 1 + 21600, service_stop_epoch=now + 20000))
    calls, observers, lock = [], [], threading.Lock()

    class Observer:
        def __init__(self, plan, label, external):
            self.plan, self.label, self.external = plan, label, external
            with lock:
                observers.append(self)

    monkeypatch.setattr(d, 'observer_class', lambda: Observer)

    def fake_run(config, path, **kwargs):
        assert kwargs['parent'] is None and isinstance(kwargs['execution_observer'], Observer)
        limiter = kwargs['request_limiter']
        assert limiter.acquire(1, timeout=600)
        limiter.reconcile(1, None)
        with lock:
            calls.append((copy.deepcopy(config), path))
        (path / 'raw/model/r000000').mkdir(parents=True)
        d.write_json(path / 'raw/model/r000000/request.json',
                     {'messages': [{'role': 'system', 'content': config['agent']['system_prompt']},
                                   {'role': 'user', 'content': 'task'}]})

    monkeypatch.setattr(d, 'l1_run', fake_run)
    monkeypatch.setattr(d.base, 'slot_record', lambda path, task, cfg: dict(
        study_valid=True, config_sha256=d.digest(cfg), path=str(path),
        usage={'model_requests_including_parent': 1}, outcome={'study_outcome_y': 0.5}))

    def build(block=0, authorization=auth):
        return d.build_block_plan(block=block, authorization=authorization, service_lease=lease,
                                  run_root=d.L1 / 'runs' / f'{d.RUN_PREFIX}{block:02d}', spec_path=spec_path)

    plan = build()
    service = tmp_path / 'service.json'
    d.write_json(service, {**plan['service_lease_origin'], 'owner_tag': plan['plan_sha256'],
                           'server_plan_hash': plan['plan_sha256'], 'leader_start_ticks': 1, 'remote_host': 'aml-b',
                           'gpu_uuid': d.GPU_UUID, 'identity_receipt_sha256': 'e' * 64,
                           'command': ['timeout', '--signal=TERM', '--kill-after=30s', '20000s',
                                       *identity['command_after_timeout']]})
    output = d.PHASE_ROOT / 'block_00' / 'execution'
    return SimpleNamespace(plan=plan, build=build, spec=spec, spec_path=spec_path, auth=auth, lease=lease,
                           service=service, output=output, source=source, base_configs=base_configs,
                           calls=calls, observers=observers, tmp=tmp_path, fake_run=fake_run)


def counts(plan):
    out = {}
    for label in plan['slots']:
        condition = plan['slot_specs'][label]['condition']
        out[condition] = out.get(condition, 0) + 1
    return out


def test_plan_is_inert_and_balanced(env):
    assert len(env.plan['slots']) == 33 and counts(env.plan) == {'O': 8, 'F': 9, 'T': 8, 'V': 8}
    assert env.plan['http_cap'] == 4620 and env.plan['logical_cap'] == 1155 and env.plan['concurrency'] == 3
    assert env.plan['d1_execution'] is False and env.plan['analyst_requests'] == 0
    assert env.calls == [] and env.observers == []
    assert not env.output.exists() and not (d.L1 / 'runs').exists()
    assert not any(label.startswith('officeval_085.') and not label.startswith('officeval_085.F')
                   for label in env.plan['slots'])


def test_conditions_change_only_the_system_prompt(env):
    header = env.spec['condition_header']
    for label in env.plan['slots']:
        row = env.plan['slot_specs'][label]
        cfg, base_cfg = env.plan['configs'][label], env.base_configs[row['task']]
        if row['condition'] == 'O':
            assert cfg == base_cfg
            continue
        before, after = copy.deepcopy(base_cfg), copy.deepcopy(cfg)
        before['agent'].pop('system_prompt'), after['agent'].pop('system_prompt')
        assert before == after
        prompt = cfg['agent']['system_prompt']
        assert prompt.count(header) == 1 and prompt.count(d.CONTRACT_HEADER) == 1
        assert prompt.index(header) < prompt.index(d.CONTRACT_HEADER)
        tail = base_cfg['agent']['system_prompt'].partition(d.CONTRACT_HEADER)[2]
        assert prompt.endswith(d.CONTRACT_HEADER + tail)
        assert d.build_execution_contract_prompt(cfg) == prompt


def test_f_text_carries_verifier_names_and_drops_internal_notes(env):
    texts = d.condition_texts(env.spec)
    for task in ('officeval_003', 'officeval_004', 'officeval_085'):
        for name in env.spec['conditions']['F']['deliverable_names'][task]['names']:
            assert name in texts[task]['F']
    assert all('待人工' not in row['F'] for row in texts.values())
    assert set(texts['officeval_085']) == {'O', 'F'}
    assert all(set(texts[t]) == set(d.CONDITIONS) for t in env.spec['tasks']['primary'])


def test_verifier_must_literally_contain_the_names(env):
    spec = copy.deepcopy(env.spec)
    entry = spec['conditions']['F']['deliverable_names']['officeval_085']
    Path(entry['verifier']['path']).write_text('no names here', encoding='utf-8')
    entry['verifier']['sha256'] = d.file_digest(Path(entry['verifier']['path']))
    with pytest.raises(ValueError, match='literally'):
        d.f_texts(spec)


def test_block_orders_cover_every_cell_once(env):
    blocks = [d.block_slots(env.spec, b) for b in range(5)]
    cells = [{label.rsplit('.', 1)[0] for label in block} for block in blocks]
    assert all(len(block) == 33 for block in blocks) and all(c == cells[0] for c in cells)
    assert len({tuple(label.rsplit('.', 1)[0] for label in block) for block in blocks}) == 5
    assert len({label for block in blocks for label in block}) == 165
    assert d.block_slots(env.spec, 2) == blocks[2]


def test_execute_runs_each_slot_once_with_its_own_observer(env):
    result = d.execute(env.plan, env.output, env.service)
    assert result['all_study_valid'] and result['slots_terminal'] == 33 and result['study_valid_count'] == 33
    assert len(env.calls) == len(env.observers) == 33
    assert len({o.external for o in env.observers}) == 33
    assert all(o.external.is_relative_to(d.PHASE_ROOT / 'block_00') for o in env.observers)
    assert result['admissions']['requests'] == 33 and result['by_condition']['F']['study_valid'] == 9
    receipt = d.read_json(env.output / ('receipt_' + env.plan['slots'][0] + '.json'))
    assert receipt['condition'] == env.plan['slot_specs'][env.plan['slots'][0]]['condition']
    assert receipt['first_request_system_sha256']
    with pytest.raises(ValueError, match='restart/overwrite'):
        d.execute(env.plan, env.output, env.service)
    assert len(env.calls) == 33


def test_invalid_slot_is_kept_without_resampling(env, monkeypatch):
    attempts = []

    def invalid(config, path, **kwargs):
        attempts.append(path)
        raise RuntimeError('fixture invalid attempt')

    monkeypatch.setattr(d, 'l1_run', invalid)
    result = d.execute(env.plan, env.output, env.service)
    assert len(attempts) == len(set(attempts)) == 33 and result['status'] == 'incomplete_or_invalid'
    rows = d.read_json(env.output / 'cohort.json')['slots']
    assert all(row['status'] == 'execution_invalid' and row['error']['traceback'] for row in rows.values())


def test_condition_prompt_mismatch_invalidates_the_slot(env, monkeypatch):
    def wrong(config, path, **kwargs):
        cfg = copy.deepcopy(config)
        cfg['agent']['system_prompt'] = 'something else'
        env.fake_run(cfg, path, **kwargs)

    monkeypatch.setattr(d, 'l1_run', wrong)
    result = d.execute(env.plan, env.output, env.service)
    assert result['study_valid_count'] == 0
    row = d.read_json(env.output / 'cohort.json')['slots'][env.plan['slots'][0]]
    assert 'system prompt differs' in row['error']['message']


@pytest.mark.parametrize('mutate', ['prompt', 'cap', 'slots', 'block'])
def test_tampered_resealed_plan_is_rejected(env, mutate):
    plan = copy.deepcopy(env.plan)
    first = plan['slots'][0]
    if mutate == 'prompt':
        plan['configs'][first]['agent']['system_prompt'] += ' extra'
    elif mutate == 'cap':
        plan['http_cap'] += 1
    elif mutate == 'slots':
        plan['slots'] = list(reversed(plan['slots']))
    else:
        plan['block'] = 1
    plan = d.seal(plan, 'plan_sha256')
    with pytest.raises(ValueError):
        d.execute(plan, env.output, env.service)
    assert env.calls == [] and not env.output.exists()


def test_authorization_binds_review_digest_and_block(env):
    value = d.read_json(env.auth)
    other = env.tmp / 'other_auth.json'
    d.write_json(other, {**value, 'authorized_blocks': [3]})
    with pytest.raises(ValueError, match='authorization'):
        env.build(authorization=other)
    d.write_json(other, {**value, 'approved_review_digest': '0' * 64})
    with pytest.raises(ValueError, match='authorization'):
        env.build(authorization=other)


@pytest.mark.parametrize('which', ['source', 'lease', 'spec'])
def test_changes_before_execution_prevent_all_sends(env, which):
    path = {'source': env.source, 'lease': env.lease, 'spec': env.spec_path}[which]
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError):
        d.execute(env.plan, env.output, env.service)
    assert env.calls == [] and not env.output.exists()


def test_output_must_stay_inside_its_block_phase(env):
    with pytest.raises(ValueError, match='inside its Exp1 block phase'):
        d.execute(env.plan, d.PHASE_ROOT / 'block_01' / 'execution', env.service)
    with pytest.raises(ValueError, match='inside its Exp1 block phase'):
        d.execute(env.plan, env.tmp / 'elsewhere', env.service)
    assert env.calls == []


def test_episode_wall_budget_denial_is_slot_local(env, tmp_path):
    caps = dict(max_requests=2, max_tokens=200, max_cost_usd=0, max_wall_seconds=1000, max_invalid_attempts=34,
                max_tokens_per_request=100, usd_per_token_upper_bound=0)
    stop = threading.Event()
    limiter = d.GuardedAdmission(d.Admission(tmp_path / 'admission.json', env.plan['plan_sha256'], caps),
                                 env.plan, {}, stop)
    assert limiter.acquire(1, timeout=599) is False and not stop.is_set()
    assert limiter.acquire(1, timeout=600) and limiter.acquire(1, timeout=600)
    assert limiter.acquire(1, timeout=600) is False and stop.is_set()


def test_service_receipt_must_match_owner_and_lease(env):
    value = d.read_json(env.service)
    value['owner_tag'] = 'f' * 64
    d.write_json(env.service, value)
    with pytest.raises(ValueError, match='service receipt'):
        d.execute(env.plan, env.output, env.service)
    assert env.calls == []


REAL = [d.ROOT / p for p in (
    'L2_Behavioral Event Localization/L2_Sandbox/validation/a_plus_i_execute_20261018/prospective_freeze.json',
    'L1_sandbox/C3_Experimental_Sandbox/third_party/OmegaUse-OfficeVal/verifiers/officeval_085_verifier.py')]


@pytest.mark.skipif(not all(p.exists() for p in REAL), reason='frozen DEV45 evidence and verifiers are local only')
def test_real_preview_keeps_o_identical_to_dev45():
    value = d.preview()
    assert value['status'] == 'preview_not_executable' and value['actor_requests'] == 0
    names = d.load_spec()[0]['conditions']['F']['deliverable_names']['officeval_085']['names']
    assert all(name in value['condition_texts']['officeval_085']['F'] for name in names)
    assert len({s for b in value['review']['slots'].values() for s in b}) == 165


# ---------------------------------------------------------------- supervisor and launcher

def test_supervisor_phase_must_be_an_exact_block_directory(tmp_path, monkeypatch):
    sup = load('supervise_acceptance_gap_exp1.py', 'exp1_test_supervisor')
    root = tmp_path / 'L2/validation/acceptance_gap_exp1_20261022'
    monkeypatch.setattr(sup.driver, 'PHASE_ROOT', root)
    phase = root / 'block_02'
    phase.mkdir(parents=True)
    names = [phase / n for n in ('plan.json', 'freeze.json', 'service.json')]
    for path in names:
        path.write_text('{}')
    assert sup.phase_paths(phase, *names) == phase.resolve()
    other = root / 'misc'
    other.mkdir()
    with pytest.raises(ValueError, match='block_NN'):
        sup.phase_paths(other, *names)
    (phase / 'execution').mkdir()
    with pytest.raises(ValueError, match='restart/overwrite'):
        sup.phase_paths(phase, *names)


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    module = load('launch_acceptance_gap_block.py', 'exp1_test_launcher')
    monkeypatch.setattr(module.driver, 'PHASE_ROOT', tmp_path / 'phase_root')
    monkeypatch.setattr(module, 'native_env', lambda spec: {})
    return module


def test_prepare_writes_integer_lease_plan_and_approval_once(launcher, tmp_path, monkeypatch):
    auth = tmp_path / 'auth.json'
    auth.write_text(json.dumps({'exact_text': '开始', 'source': 'user'}))
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return {'plan_sha256': 'f' * 64, 'review_digest': 'r' * 64, 'experiment_id': 'acceptance_gap_exp1_20261022'}

    monkeypatch.setattr(launcher.driver, 'build_block_plan', fake_build)
    out = launcher.prepare(1, auth, now=1792600000.75)
    phase = launcher.phase_dir(1)
    lease = json.loads((phase / 'lease.json').read_text())
    assert lease['started_epoch'] == 1792600000.0 and lease['hard_stop_epoch'] - lease['started_epoch'] == 21600
    assert lease['service_stop_epoch'] == lease['hard_stop_epoch'] - 600
    freeze = json.loads((phase / 'freeze.json').read_text())
    assert freeze['approved_plan_sha256'] == 'f' * 64 and freeze['block'] == 1 and freeze['d0_frozen'] is False
    assert freeze['user_authorization_record']['exact_text'] == '开始'
    assert seen['run_root'].name == 'acceptance_gap_exp1_20261022_b01' and out['plan_sha256'] == 'f' * 64
    with pytest.raises(ValueError, match='never re-prepared'):
        launcher.prepare(1, auth)


def test_failed_prepare_leaves_nothing_behind(launcher, tmp_path, monkeypatch):
    auth = tmp_path / 'auth.json'
    auth.write_text('{}')

    def failing(**kwargs):
        raise ValueError('authorization missing')

    monkeypatch.setattr(launcher.driver, 'build_block_plan', failing)
    with pytest.raises(ValueError, match='authorization'):
        launcher.prepare(0, auth)
    assert not launcher.phase_dir(0).exists()


def test_start_service_binds_payload_to_lease_and_plan(launcher, tmp_path, monkeypatch):
    phase = launcher.phase_dir(0)
    phase.mkdir(parents=True)
    plan = {'plan_sha256': 'a' * 64}
    lease = {'started_epoch': 1000.0, 'hard_stop_epoch': 22600.0, 'service_stop_epoch': 22000.0}
    (phase / 'plan.json').write_text(json.dumps(plan))
    (phase / 'lease.json').write_text(json.dumps(lease))
    monkeypatch.setattr(launcher.driver, 'validate_plan', lambda value: None)
    checked = []
    monkeypatch.setattr(launcher.driver, 'validate_service', lambda p, path: checked.append(path))
    expected = launcher.driver.file_digest(launcher.REMOTE_LAUNCHER_LOCAL)
    sent = []

    def runner(args, input=None, **kwargs):
        sent.append((args[-1], input))
        if 'hashlib' in args[-1]:
            return SimpleNamespace(returncode=0, stdout=expected + '\n', stderr='')
        if 'time.time' in args[-1]:
            return SimpleNamespace(returncode=0, stdout='1234.5\n', stderr='')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'leader_pid': 7}) + '\n', stderr='')

    assert launcher.start_service(0, execute=False, runner=runner)['status'] == 'not_executed' and not sent
    launcher.start_service(0, execute=True, runner=runner, clock=iter([2000.0, 2000.2]).__next__)
    payload = json.loads(sent[-1][1])
    assert payload['plan_sha256'] == 'a' * 64 and payload['local_start_epoch'] == 1000.0
    assert payload['service_stop_epoch'] == 22000.0 and payload['clock_calibration']['remote_epoch'] == 1234.5
    assert json.loads((phase / 'service.json').read_text()) == {'leader_pid': 7} and checked
    with pytest.raises(ValueError, match='already attempted'):
        launcher.start_service(0, execute=True, runner=runner)


def test_start_service_refuses_a_different_remote_launcher(launcher, monkeypatch):
    phase = launcher.phase_dir(0)
    phase.mkdir(parents=True)
    (phase / 'plan.json').write_text(json.dumps({'plan_sha256': 'a' * 64}))
    (phase / 'lease.json').write_text(json.dumps({'started_epoch': 1.0, 'service_stop_epoch': 2.0}))
    monkeypatch.setattr(launcher.driver, 'validate_plan', lambda value: None)

    def runner(args, input=None, **kwargs):
        return SimpleNamespace(returncode=0, stdout='0' * 64 + '\n', stderr='')

    with pytest.raises(ValueError, match='remote service launcher differs'):
        launcher.start_service(0, execute=True, runner=runner)
    assert not (phase / 'service_start_attempt.json').exists()


def test_readiness_timeout_does_not_release_or_mark_failure(launcher, monkeypatch):
    phase = launcher.phase_dir(0)
    phase.mkdir(parents=True)
    for name in ('plan.json', 'freeze.json', 'service.json'):
        (phase / name).write_text(json.dumps({'plan_sha256': 'a' * 64}))
    monkeypatch.setattr(launcher, 'READY_SECONDS', 0.01)

    def opener(*args, **kwargs):
        raise OSError('no tunnel')

    assert launcher.launch(0, execute=False)['status'] == 'not_executed'
    with pytest.raises(TimeoutError):
        launcher.launch(0, execute=True, opener=opener)
    assert not (phase / 'startup_failed.json').exists()


# ---------------------------------------------------------------- analysis

ANALYSIS_SPEC = None


def spec_analysis():
    global ANALYSIS_SPEC
    if ANALYSIS_SPEC is None:
        ANALYSIS_SPEC = an.load_spec()[0]
    return ANALYSIS_SPEC


def synthetic(means, blocks=5, noise=0.05, seed=1, drop=None):
    spec = spec_analysis()
    rng = random.Random(seed)
    runs = []
    for block in range(blocks):
        for task in spec['tasks']['primary']:
            for condition in ('O', 'F', 'T', 'V'):
                y = max(0.0, min(1.0, means[condition] + rng.uniform(-noise, noise)))
                runs.append(dict(block=block, label=f'{task}.{condition}.r{block:02d}', task=task, condition=condition,
                                 status='completed', valid=True, y=y, dim1=True, max_score=10,
                                 items=[dict(rule='keep', delta=1, max_delta=1)], requests=20))
        runs.append(dict(block=block, label=f'officeval_085.F.r{block:02d}', task='officeval_085', condition='F',
                         status='completed', valid=True, y=0.3, dim1=True, max_score=10, items=[], requests=25))
    if drop:
        runs = [r for r in runs if not drop(r)]
    return [dict(block=b) for b in range(blocks)], runs


def decision(means, **kwargs):
    blocks, runs = synthetic(means, **kwargs)
    return an.analyze(blocks, runs, spec_analysis(), 'x' * 64)


@pytest.mark.parametrize('means,expected', [
    ({'O': 0.2, 'F': 0.6, 'T': 0.25, 'V': 0.22}, 'go_method_development'),
    ({'O': 0.2, 'F': 0.25, 'T': 0.2, 'V': 0.2}, 'no_go_information_not_binding'),
    ({'O': 0.2, 'F': 0.6, 'T': 0.55, 'V': 0.3}, 'stop_existing_methods_sufficient'),
    ({'O': 0.2, 'F': 0.6, 'T': 0.4, 'V': 0.3}, 'partial_requires_user_decision'),
])
def test_final_decision_rules(means, expected):
    summary = decision(means)
    assert summary['decision'] == expected
    assert summary['actor_requests'] == 0 and summary['f_only']['valid'] == 5


def test_interim_futility_and_continue():
    flat = {'O': 0.2, 'F': 0.2, 'T': 0.2, 'V': 0.2}
    assert decision(flat, blocks=3, noise=0.01)['decision'] == 'stop_futility'
    assert decision(flat, blocks=2, noise=0.01)['decision'] == 'interim_continue'


def test_final_requires_four_valid_runs_per_primary_o_and_f_cell():
    summary = decision({'O': 0.2, 'F': 0.6, 'T': 0.25, 'V': 0.22},
                       drop=lambda r: r['task'] == 'officeval_003' and r['condition'] == 'O' and r['block'] < 2)
    assert summary['decision'] == 'insufficient_data'


def test_bootstrap_is_deterministic_for_the_frozen_seed():
    a = decision({'O': 0.2, 'F': 0.6, 'T': 0.25, 'V': 0.22})
    b = decision({'O': 0.2, 'F': 0.6, 'T': 0.25, 'V': 0.22})
    assert a['intervals_ci95'] == b['intervals_ci95'] and a['artifact_sha256'] == b['artifact_sha256']


def test_harm_flags_conditions_that_break_items_o_gets_right():
    blocks, runs = synthetic({'O': 0.2, 'F': 0.6, 'T': 0.25, 'V': 0.22})
    for run in runs:
        if run['condition'] == 'T':
            run['items'] = [dict(rule='keep', delta=0, max_delta=1)]
    report = an.harm(runs, spec_analysis()['tasks']['primary'], ('F', 'T', 'V'), spec_analysis()['analysis']['harm'])
    assert report['T']['harmed'] == 8 and report['F']['harmed'] == 0


def test_source_deductions_split_gate_and_rubric_sources():
    rule = spec_analysis()['analysis']['source_rule']
    runs = [dict(task='officeval_003', condition='O', valid=True, y=0.0, dim1=False, max_score=15, items=[]),
            dict(task='officeval_003', condition='O', valid=True, y=0.5, dim1=True, max_score=10,
                 items=[dict(rule='文档表格中没有空白行', delta=0, max_delta=5),
                        dict(rule='other', delta=0, max_delta=5)])]
    out = an.source_deductions(runs, ['officeval_003'], ('O',), rule)['O']
    assert out['gate_or_no_delivery'] == pytest.approx(0.5)
    assert out['written'] == pytest.approx(0.25) and out['shown'] == pytest.approx(0.25)


def test_drift_against_reference_o():
    tasks = spec_analysis()['tasks']['primary']
    values = {(t, 'O'): [0.3] for t in tasks}
    reference = [dict(task=t, y=0.1) for t in tasks]
    report = an.drift(values, reference, tasks, 0.10)
    assert report['task_equal_difference'] == pytest.approx(0.2) and report['flagged'] is True


def fake_block_phase(root, block, spec_sha, means):
    """Minimal sealed block outputs in the driver's layout (executed plan, cohort, graded packages)."""
    spec = spec_analysis()
    labels = [f'{t}.{c}.r{block:02d}' for t in spec['tasks']['primary'] for c in ('O', 'F', 'T', 'V')]
    labels.append(f'officeval_085.F.r{block:02d}')
    execution = root / f'block_{block:02d}' / 'execution'
    execution.mkdir(parents=True)
    plan = an.seal(dict(spec_sha256=spec_sha, block=block, slots=labels), 'plan_sha256')
    (execution / 'executed_plan.json').write_text(json.dumps(plan))
    slots = {}
    for index, label in enumerate(labels):
        task, condition, _ = an.parse_label(label)
        package = root / 'packages' / label
        package.mkdir(parents=True)
        (package / 'official_grade.json').write_text(json.dumps({'official_raw': {
            'dim1_pass': True, 'max_score': 10, 'dim2_items': [dict(rule='keep', delta=1, max_delta=1)]}}))
        y = means.get(condition, 0.3) + 0.01 * (index % 3)
        slots[label] = dict(status='completed', study_valid=True, path=str(package),
                            outcome=dict(study_valid=True, study_outcome_y=y),
                            usage=dict(model_requests_including_parent=20))
    (execution / 'cohort.json').write_text(json.dumps(an.seal(dict(plan_sha256=plan['plan_sha256'], slots=slots))))
    return root / f'block_{block:02d}'


def test_exported_inputs_reproduce_the_direct_analysis(tmp_path, monkeypatch):
    spec, spec_sha = an.load_spec()
    reference = [dict(task=t, y=0.2) for t in spec['tasks']['primary']]
    monkeypatch.setattr(an, 'load_reference', lambda value: reference)
    means = {'O': 0.2, 'F': 0.6, 'T': 0.25, 'V': 0.22}
    phases = [fake_block_phase(tmp_path, b, spec_sha, means) for b in range(3)]
    inputs = an.export_inputs(phases)
    assert inputs['evidence_reverified'] is False and len(inputs['runs']) == 99
    remote = an.analyze_inputs(json.loads(json.dumps(inputs)), spec, spec_sha)
    blocks, runs = [], []
    for phase in phases:
        meta, block_runs = an.load_block(phase, spec_sha, reverify=False)
        blocks.append(meta)
        runs += block_runs
    direct = an.analyze(blocks, runs, spec, spec_sha, reference)
    assert remote['decision'] == direct['decision'] == 'interim_continue'
    assert remote['estimates'] == direct['estimates'] and remote['intervals_ci95'] == direct['intervals_ci95']
    assert remote['provenance']['inputs_sha256'] == inputs['artifact_sha256']
    tampered = json.loads(json.dumps(inputs))
    tampered['runs'][0]['y'] = 1.0
    with pytest.raises(ValueError, match='seal'):
        an.analyze_inputs(tampered, spec, spec_sha)
    with pytest.raises(ValueError, match='not exported under this spec'):
        an.analyze_inputs(inputs, spec, '0' * 64)
