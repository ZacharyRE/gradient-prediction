"""Record terminal supplement review after inspecting actual analyses and warning cases."""
from common import *
paths=['audits/expanded_unaveraged_supplement_plan.json','audits/expanded_unaveraged_supplement_frozen.json','audits/expanded_unaveraged_supplement_status.json','results/expanded_unaveraged_supplement_analysis.json','audits/supplement_scoring_warning_review.json','audits/expanded_supplement_gpu2_protocol_replay.json','audits/expanded_supplement_gpu1_protocol_replay.json','audits/balanced_ood_execution.json','audits/balanced_ood_zombie_read_amendment.json','audits/balanced_gpu2_existing_replay_reuse.json']
a={p:json.loads((S/p).read_text()) for p in paths};plan=a[paths[0]];ph=sha(S/paths[0]);frozen=a[paths[1]];status=a[paths[2]];result=a[paths[3]];warnings=a[paths[4]];queue=a[paths[7]]
assert frozen['plan_sha256']==status['plan_sha256']==result['plan_sha256']==warnings['plan_sha256']==ph
assert status['all_complete'] and queue['all_complete'] and not queue['running'] and not queue['errors']
assert warnings['all_reviewed'] and warnings['n_warning_events']==2
assert result['script_sha256']==sha(S/'scripts/analyze_expanded_supplement.py')==frozen['analysis_sha256']
assert result['frozen_sha256']==sha(S/paths[1])
for p in paths[5:7]:assert a[p]['passed']
assert a[paths[5]]['actual_gpu']=='2' and a[paths[5]]['n']==1000
assert a[paths[9]]['passed'] and a[paths[9]]['actual_replay_sha256']==sha(S/paths[5])
assert status['balanced_execution_amendment_sha256']==sha(S/paths[8])==queue['execution_amendment_sha256']
validated=0
for ds,models in result['identities'].items():
 for name,ids in models.items():
  for fn,key in [('math_predictions.jsonl','predictions_sha256'),('math_manifest.json','manifest_sha256')]:assert sha(S/'results/evaluation'/ds/name/fn)==ids[key]
  if name in plan['models'].values():validated+=1
assert validated==25
for name,v in frozen['artifacts'].items():
 for filename,digest in v['files'].items():assert sha(Path(v['adapter'])/filename)==digest
extra='teacher7_expanded_epoch4'
conclusions={m:dict(math=v['datasets']['math_reused5000']['families'][extra],ood_macro=v['ood_macro'][extra],descriptive_criteria=v['descriptive_criteria']) for m,v in result['results'].items()}
write(S/'audits/expanded_supplement_final_review.json',dict(time=time.time(),all_reviewed=True,gpu_work_terminal=True,plan_sha256=ph,all25_supplement_pairs_validated=True,actual_gpus=status['actual_gpus'],source_sha256={p:sha(S/p) for p in paths},script_sha256=sha(__file__),conclusions=conclusions,scope='Actual supplementary terminal execution, allfivefrozenweights,25evaluation pairs, shared21modeldata hashes, numeric scoring, and bothwarningfulltextreviews checked. OriginalCPUhandoffreturncodes preserved by separatemainreconciliation. RawMATH hasseed47belowbase and allfive rawOODmacrodeltasnegative; strictOODpositive is format-sensitive and doesnotreplaceprimary. No claimallmodelreasoning reviewed; fourfamilies exploratoryfactorial, not causalcoverage-only experiment.'))
print('Supplement25pairs, all5weights, actualGPU2replay and2warning reviews validated.')
