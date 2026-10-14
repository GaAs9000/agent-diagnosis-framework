"""Offline statistical contracts; no model calls or Docker execution."""
import random
import pytest
from l2_sandbox import CODE_VERSION
from l2_sandbox.curve import infer_curve_from_samples
from l2_sandbox.io import L2Error, code_fingerprint, write_json
from l2_sandbox.predicates import RULE_VOCABULARY
from l2_sandbox.score import evaluate_hypotheses, score_episode, score_values, summarize_runs
from l2_sandbox.stats import clustered_bootstrap, fisher_exact_one_sided, holm_adjust, lower_drop, mantel_haenszel_pool, permutation_one_sided, upper


@pytest.mark.parametrize('values', [None, {}, 'bad', [], [.8], [.8, None], [.8, float('nan')], [.8, float('inf')], [.8, 2], [.8, True], [.8, .2, .1]])
def test_partial_invalid_predictions_are_unscored(values):
    result = score_values(values, [{'valid': True, 'k': 0, 'y': .8}], 2, .8, 0)
    assert result['status'] == 'unscored'
    assert result['brier'] is result['linear_decay_brier'] is result['skill_score'] is None


def test_score_only_active_current_bound_outputs(tmp_path):
    write_json(tmp_path/'plan.json', {'episode_id': 'ep', 'turns': 2, 'terminal_outcome': 0})
    binding = {'parent_index_sha256': 'parent', 'payload_sha256': 'input'}
    doc = {'analyst': 'rules@1', 'episode_id': 'ep', 'code_version': CODE_VERSION,
           'code_fingerprint': code_fingerprint(), 'input_sha256': binding, 'values': [.8, .4]}
    write_json(tmp_path/'values/rules.json', doc)
    write_json(tmp_path/'values/inactive.json', {**doc, 'analyst': 'llm:old', 'episode_id': 'other'})
    kwargs = dict(active_analysts=['rules@1'], expected_input_bindings={'values': {'rules@1': binding}})
    result = score_episode(tmp_path, **kwargs)
    assert list(result['analysts']) == ['rules@1']
    assert result['analysts']['rules@1']['cost']['basis'] == 'configured_estimate'
    for field, wrong in [('episode_id', 'other'), ('input_sha256', {'parent_index_sha256': 'other'}), ('code_fingerprint', 'old')]:
        write_json(tmp_path/'values/rules.json', {**doc, field: wrong})
        with pytest.raises(L2Error, match='Stale or unbound'):
            score_episode(tmp_path, **kwargs)
    (tmp_path/'values/rules.json').unlink()
    assert score_episode(tmp_path, **kwargs)['analysts']['rules@1']['value_prediction']['status'] == 'unscored'


def hypothesis_case(stage='s2'):
    parent = [{'turn': 2, 'type': 'tool_exit', 'value': {'ok': False}}]
    rows = [{'stage': stage, 'k': 1, 'trial': str(i), 'valid': True, 'y': float(i >= 4)} for i in range(8)]
    facts = {str(i): [{'turn': 2, 'type': 'tool_exit', 'value': {'ok': i >= 4}}] for i in range(8)}
    hypotheses = [{'turn': 2, 'predicate': RULE_VOCABULARY['tool_execution_anomaly'], 'effect': -.5, 'rationale': 'anomaly'}]
    curve = {'verdict': 'concentrated_turn', 'turn': 2, 'turn_drops': {'2': .5}}
    return hypotheses, parent, curve, facts, rows


def test_hypotheses_stage2_only_missing_facts_not_negative_predicate():
    args = hypothesis_case('s1')
    assert evaluate_hypotheses(*args)['hypotheses'][0]['status'] == 'untestable'
    args = hypothesis_case()
    assert evaluate_hypotheses(*args)['hypotheses'][0]['status'] == 'confirmed'
    args[3].pop('7')
    evaluation = evaluate_hypotheses(*args)
    assert evaluation['explanation_covered'] is None
    result = evaluation['hypotheses'][0]
    assert result['status'] == 'untestable' and result['reason'] == 'missing_branch_facts'


def entry(brier, coverage, baseline=.5):
    return {'value_prediction': {'brier': brier, 'linear_decay_brier': baseline},
            'hypothesis_testing': {'proposed_count': 1, 'confirmed_count': int(bool(coverage)), 'explanation_covered': coverage},
            'cost': {'equivalent_continuations': 0}}


def save_case(root, name, family, rule, llm=None):
    path=root/name
    analysts={'rules@1': rule}
    if llm is not None: analysts['llm:C']={**llm, 'analyst_group': 'C'}
    write_json(path/'scores.json', {'episode_id': name, 'family_id': family, 'verdict': 'concentrated_turn', 'analysts': analysts})
    return path


def test_metric_eligible_families_and_same_skill_denominator(tmp_path):
    a=save_case(tmp_path,'a','f1',entry(.2,True))
    b=save_case(tmp_path,'b','f2',entry(None,None,baseline=100))
    summary=summarize_runs([a,b])['analysts']['rules@1']
    assert summary['skill_score']['estimate'] == pytest.approx(.6)
    assert summary['skill_score']['family_count'] == summary['explanation_coverage']['family_count'] == 1
    assert summary['skill_score']['ci_95'] == [None,None]
    zero=save_case(tmp_path,'zero','f3',entry(0,None,baseline=0))
    assert summarize_runs([a,zero])['analysts']['rules@1']['skill_score']['family_count']==1


def test_hybrid_requires_paired_both_positive_holdout_bounds(tmp_path):
    paths=[save_case(tmp_path,f'e{i}',f'f{i}',entry(.5,False),entry(.1,True)) for i in range(2)]
    default=summarize_runs(paths)
    assert default['recommended_default_mode']=='program_only' and default['hybrid_verification_status']=='not_verified'
    verified=summarize_runs(paths,heldout_families=['f0','f1'])
    assert verified['recommended_default_mode']=='hybrid'
    comparison=verified['paired_holdout_comparisons']['llm:C']
    assert comparison['skill_difference']['ci_95'][0]>0 and comparison['explanation_coverage_difference']['ci_95'][0]>0
    rawpaths=[]
    for i in range(2):
        path=tmp_path/f'raw{i}'
        write_json(path/'scores.json',{'episode_id':f'raw{i}','family_id':f'f{i}','verdict':'concentrated_turn',
                   'analysts':{'rules@1':entry(.5,False),'llm:B':{**entry(.1,True),'analyst_group':'B'}}})
        rawpaths.append(path)
    assert summarize_runs(rawpaths,heldout_families=['f0','f1'])['recommended_default_mode']=='program_only'
    # Exact zero rule skill is a real value; one improvement alone is insufficient.
    paths=[save_case(tmp_path,f'e{i}',f'f{i}',entry(.5,False),entry(.1,False)) for i in range(2)]
    assert summarize_runs(paths,heldout_families=['f0','f1'])['recommended_default_mode']=='program_only'
    # Marginally excellent unpaired LLM episodes cannot prove a paired gain.
    paths=[save_case(tmp_path,'rule','f0',entry(.5,False))]
    p=tmp_path/'llmonly'; write_json(p/'scores.json',{'episode_id':'llmonly','family_id':'f1','verdict':'concentrated_turn','analysts':{'llm:C':entry(0,True)}})
    assert summarize_runs([*paths,p],heldout_families=['f0','f1'])['recommended_default_mode']=='program_only'


def test_mh_small_strata_are_descriptive_not_fabricated_precision():
    small={'n_p':1,'n_not_p':1,'mean_p':0,'mean_not_p':1,'family_id':'f1','y_p':[0],'y_not_p':[1]}
    result=mantel_haenszel_pool([small])
    assert result['pooled_effect']==-1 and result['ci_95']==[None,None] and result['p_value'] is None
    supported=[dict(small,n_p=4,n_not_p=4,y_p=[0]*4,y_not_p=[1]*4,family_id=f'f{i}') for i in range(2)]
    assert mantel_haenszel_pool(supported)['status']=='evaluated'


def test_fisher_holm_and_permutation_order_invariance():
    with pytest.raises(L2Error, match='requires binary'):
        fisher_exact_one_sided([.2]*4,[.8]*4)
    assert fisher_exact_one_sided([0]*4,[1]*4)==pytest.approx(1/70)
    assert holm_adjust([.01,.03,.2])==pytest.approx([.03,.06,.2])
    p=[i/30 for i in range(12)]; q=[.6+i/40 for i in range(12)]
    assert permutation_one_sided(p,q)==permutation_one_sided(p[::-1],q[::-1])
    h,parent,curve,facts,rows=hypothesis_case()
    assert evaluate_hypotheses(h,parent,curve,facts,rows)==evaluate_hypotheses(h,parent,curve,facts,list(reversed(rows)))
    continuous=[.1,.4,.6,.9,.3,.2]
    assert upper(continuous)==upper(continuous[::-1])
    assert lower_drop(continuous,[.1,.2,.3])==lower_drop(continuous[::-1],[.3,.2,.1])


def test_bounded_continuous_and_holm_null_calibration():
    rng=random.Random(20261014)
    upper_misses=drop_fp=holm_fp=perm_fp=0
    for trial in range(1000):
        a=[rng.random() for _ in range(24)]; b=[rng.random() for _ in range(24)]
        if trial < 120:
            upper_misses+=upper(a)<.5
            drop_fp+=lower_drop(a,b)>0
        pvals=[permutation_one_sided([rng.random() for _ in range(6)],[rng.random() for _ in range(6)]) for _ in range(4)]
        perm_fp+=sum(p<=.05 for p in pvals)
        holm_fp+=any(p<=.05 for p in holm_adjust(pvals))
    print(f'continuous/Holm calibration: upper misses={upper_misses}/120, drop FP={drop_fp}/120, Holm FWER={holm_fp}/1000, permutation FP={perm_fp}/4000')
    # Bounded offline checks, not a universal finite-sample coverage theorem.
    # Percentile-upper coverage is diagnostic only, not a certified nominal 95% bound.
    assert drop_fp/120<=.07
    assert holm_fp/1000<=.06
    assert perm_fp/4000<=.06


def test_localization_and_hypothesis_permutation_invariance():
    s1={0:[1.0]*12,4:[0.0]*12}
    s2={0:[1.0]*24,1:[1.0]*24,2:[1.0]*24,3:[0.0]*24,4:[0.0]*24}
    curve=infer_curve_from_samples(8,0,[0,4,8],s1,s2)
    assert curve['verdict']=='concentrated_turn' and curve['turn']==3
    # Localizer sees checkpoint marginals only; within-checkpoint row order cannot alter it.
    permuted=infer_curve_from_samples(8,0,[0,4,8],{k:list(reversed(v)) for k,v in s1.items()},
                                      {k:list(reversed(v)) for k,v in s2.items()})
    assert curve==permuted
    args=list(hypothesis_case())
    args[0][0]['turn']=3; args[1][0]['turn']=3; args[2]=curve
    for facts in args[3].values(): facts[0]['turn']=3
    for row in args[4]: row['k']=2
    first=evaluate_hypotheses(*args)
    rng=random.Random(20261014); rng.shuffle(args[4])
    assert first==evaluate_hypotheses(*args)


def test_bounded_continuous_curve_null_and_alternative():
    rng=random.Random(20261014)
    null_fp=power=0
    trials=50
    def draw(v,n):
        # Symmetric bounded uniforms with exact population mean v.
        return [rng.uniform(max(0,2*v-1), min(1,2*v)) for _ in range(n)]
    for alternative in (False,True):
        for trial in range(trials):
            j=trial%16+1
            values={0:.7}
            for k in range(1,17):
                drop=(.5 if k==j else .2/15) if alternative else .7/16
                values[k]=max(0,values[k-1]-drop)
            s1={k:draw(values[k],12) for k in [0,4,8,12]}
            result=infer_curve_from_samples(16,0,[0,4,8,12,16],s1,
                     lambda ks:{k:draw(values[k],24) for k in ks})
            if alternative:
                power+=result['verdict']=='concentrated_turn' and result['turn']==j
            else:
                null_fp+=result['verdict'].startswith('concentrated_')
    print(f'bounded continuous curve diagnostic: null FP={null_fp}/{trials}, correctturn power={power}/{trials}')
    assert null_fp/trials<=.07
    assert power/trials>=.65
