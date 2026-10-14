import pytest
from c3_sandbox.integrated.experiment_card import *

def card(operator='original'):
    return ExperimentCard.create(
        phenomenon='visibility affects repair', evidence_references=['ev-1'],
        competing_explanations=[{'explanation_id':'a','description':'context','probability':.2}, {'explanation_id':'b','description':'feedback','probability':.2}],
        candidate_experiments=[{'experiment_id':'exp-1','condition':{'visibility':'compact'},'operator':operator,'checkpoint':{'checkpoint_id':'cp-1'},'control':{'kind':'matched'},'endpoint':'score','budget':3,'applicability':'E2-S'}])

def test_missing_evidence_and_operator_rejection():
    compiler=ExperimentCardCompiler()
    with pytest.raises(EvidenceMissingError): compiler.compile(card(), evidence_ids=[], system_identity='s', runtime_identity='r')
    with pytest.raises(OperatorRejectedError): compiler.compile(card('free_script'), evidence_ids=['ev-1'], system_identity='s', runtime_identity='r')

def test_probabilities_need_not_sum_one():
    assert sum(x.probability for x in card().competing_explanations) == .4

def test_versions_are_immutable_and_chained():
    old=card(); new=old.seal_predictions()
    assert old.version == 1 and new.version == 2 and new.parent_card_id == old.card_id
    assert old.status is CardStatus.PENDING

def test_compile_carries_identity_and_budget():
    branch=ExperimentCardCompiler().compile(card(), evidence={'ev-1':{}}, system_identity={'model':'m'}, runtime_identity={'image':'sha256:x'})[0]
    assert branch['checkpoint_identity'] and branch['probe_identity'] and branch['system_identity']['model']=='m'
    assert branch['phase']=='exploration' and branch['budget']==3 and branch['endpoint']=='score'

def test_prediction_sealing_and_authorized_update():
    sealed=card().seal_predictions()
    with pytest.raises(UnauthorizedRevealError): sealed.update_after_reveal({'score':1})
    updated=sealed.update_after_reveal({'score':1}, authorized_reveal=True, explanation_updates={'a': {'probability': .8}})
    assert updated.status is CardStatus.OBSERVED and updated.parent_card_id == sealed.card_id
    assert updated.competing_explanations[0].probability == .8

def test_explicit_outcomes():
    assert outcome_not_run()['status']=='NOT_RUN'
    assert outcome_selector_invalid()['status']=='selector_invalid'
    assert outcome_unknown()['status']=='unknown'
