"""Generate the distributed frozen L2 JSON Schema. Version schema on changes."""
import json
from pathlib import Path
S={'type':'string'}; ARR={'type':'array'}; OBJ={'type':'object'}
def enum(*v):return {'enum':list(v)}
def array(item=S):return {'type':'array','items':item}
def obj(fields,required=None):
    return {'type':'object','additionalProperties':False,'properties':fields,
            'required':list(fields) if required is None else list(required)}
sha={'type':'string','pattern':'^[0-9a-f]{64}$'}
cp_id={'type':'string','pattern':'^cp[0-9]{6}$'}
prob={'type':'number','minimum':0.0,'maximum':1.0}
strata_enum=enum('tool_execution_anomaly','submission_modified','assistant_finished_turn','submission_readback','object_structure_changed',None)

fact=obj({'fact_id':S,'fact_type':S,'origin':enum('recorded','derived'),'availability':enum('present','unknown','unsupported','invalid'),
          'value':{},'source_ref_ids':{'type':'array','items':S,'minItems':1,'uniqueItems':True},'event_ids':array(),
          'scope':OBJ,'derivation':{'type':['object','null']},'coverage':OBJ})
local=obj({'localization_id':S,'category':enum('protocol_termination','tool_execution_anomaly','submission_missing','final_score_gap',
 'file_version_change','restoration_observed','state_stability','object_structure_attention','other_evidence_bounded_event',
 'concentrated_turn','concentrated_interval','diffuse_loss','out_of_reach'),
 'granularity':enum('event','event_set','interval','terminal'),'event_ids':{'type':'array','items':S,'uniqueItems':True},
 'statement':S,'object_scope':OBJ,'before_version':{'type':['string','null']},'after_version':{'type':['string','null']},
 'supporting_fact_ids':{'type':'array','items':S,'minItems':1,'uniqueItems':True},'counterevidence_fact_ids':array(),
 'task_relation':obj({'requirement_fact_id':{'type':['string','null']},'interpretation':S,'origin':enum('analyst_interpretation','recorded')}),
 'uncertainty':array()})
prediction=obj({'intervention_template':enum('provide_public_state_summary','generic_reminder','resume_unchanged','value_curve'),
 'control_template':enum('resume_unchanged','generic_reminder'),'direction':enum('increase','decrease','no_change','unknown'),
 'proximal_prediction':S,'endpoint_prediction':S,'uncertainty':S})
hyp=obj({'hypothesis_id':S,'localization_ids':{'type':'array','items':S,'minItems':1},'mechanism_statement':S,
 'supporting_fact_ids':{'type':'array','items':S,'minItems':1},'counterevidence_fact_ids':array(),
 'alternatives':{'type':'array','items':S,'minItems':1},'assumptions':array(),'unobserved_requirements':array(),
 'predictions':array(prediction),'target_behavior':strata_enum,
 'abstention_reason':{'type':['string','null']},'verdict':{'const':'untested'},
 'scope':{'const':'this_episode_and_explicit_future_protocol_only'}},
 required=['hypothesis_id','localization_ids','mechanism_statement','supporting_fact_ids','counterevidence_fact_ids',
           'alternatives','assumptions','unobserved_requirements','predictions','abstention_reason','verdict','scope'])
loss_interval=obj({'start_checkpoint':cp_id,'end_checkpoint':cp_id,
 'turn':{'type':['integer','null'],'minimum':1},'confidence':prob},
 required=['start_checkpoint','end_checkpoint','confidence'])
diagnosis=obj({'summary':S,'localizations':array(local),'hypotheses':array(hyp),'abstentions':array(),
 'value_predictions':{'type':'object','patternProperties':{'^cp[0-9]{6}$':prob},'additionalProperties':False},
 'predicted_loss_intervals':{'type':'array','items':loss_interval,'maxItems':3}},
 required=['summary','localizations','hypotheses','abstentions'])
manifest=obj({'schema_version':{'const':'l2-0.1.0'},'package_kind':enum('diagnosis','experiment_plan','experiment_execution','experiment_validation','value_curve_plan','value_curve_execution','value_curve_results'),
 'analysis_id':S,'created_at':S,'producer':OBJ,'source_index_sha256':sha,'source_episode_id':S,
 'mode':S,'status':S,'failure':{'type':['object','null']},'counts':OBJ,'claim_guarantees':OBJ,'source_unchanged':{'type':'boolean'},'config_sha256':sha})
index=obj({'schema_version':{'const':'l2-0.1.0'},'package_kind':S,'files':array(obj({'path':S,'sha256':sha,'size_bytes':{'type':'integer','minimum':0}})),
 'trust_scope':S})
root={'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:l2:0.1.0', '$defs':{'fact':fact,'localization':local,'hypothesis':hyp,'diagnosis':diagnosis,'manifest':manifest,'index':index}}
p=Path(__file__).resolve().parents[1]/'src/l2_sandbox/schemas/contracts.json';p.write_text(json.dumps(root,indent=2)+'\n')
