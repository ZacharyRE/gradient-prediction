from common import *
from gradient_geometry.sft_protocol import artifact_identity
p=S/'audits/long_decode_plan.json'
assert not p.exists()
models={'base':None,'sample_all_lr1e5_s43_epoch2':str(S/'results/training/sample_all_lr1e5_s43/epoch2'),'teacher_all_lr5e5_s43_epoch4':str(S/'results/training/teacher_all_lr5e5_s43/epoch4')}
write(p,dict(time=time.time(),models=models,adapter_files_sha256={k:artifact_identity(Path(v))['files'] for k,v in models.items() if v},data_sha256=sha(S/'data/dev.jsonl'),max_tokens=4096,purpose='Equal-budget development test of whether2048-token termination suppresses SFT benefits; frozen two candidate checkpoints, not best chosen after long-budget results. Main originalgreedy results remain2048; no OOD.'))
rows={}
for dataset in ['sample_all','teacher_all','teacher32_all']:
 for r in read(S/f'data/{dataset}.jsonl'):
  key=r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest()
  if key not in rows:rows[key]=dict(r,target_sha256=key[1],audit_datasets=[dataset])
  else:rows[key]['audit_datasets'].append(dataset)
for r in read(S/'audits/nll_noguided_review.jsonl')+read(S/'audits/nll_replacement_review.jsonl')+read(S/'audits/nll_replacement_review2.jsonl'):
 key=r['original_index'],r['target_sha256']
 if key not in rows:rows[key]=dict(r,audit_datasets=['nll_review_calibration'])
jsonl(S/'data/judge_selected_sources_input.jsonl',list(rows.values()));write(S/'audits/judge_selected_sources_plan.json',dict(time=time.time(),n=len(rows),input_sha256=sha(S/'data/judge_selected_sources_input.jsonl'),purpose='Weak process-quality audit of actual selected targets and independent followup calibration on harder NLL review counterexamples. No automatic quality-certified label, training-data-only.'))
p=S/'audits/sampling_seed_stability_plan.json';plan=json.loads(p.read_text());plan['after_sampling_tasks']=[dict(name='long_decode_probe',command=[sys.executable,str(S/'scripts/evaluate_long_probe.py')],log=str(S/'logs/long_decode_probe.log')),dict(name='selected_source_judge',command=[sys.executable,str(S/'scripts/judge_targets.py'),'--input',str(S/'data/judge_selected_sources_input.jsonl'),'--name','selected_sources'],log=str(S/'logs/judge_selected_sources.log'))];write(p,plan)
p=S/'scripts/gpu_worker_registry.json';reg=json.loads(p.read_text());write(p,sorted(set(reg+['evaluate_long_probe.py'])))
print(json.dumps(dict(judge_n=len(rows),next_tasks=[t['name'] for t in plan['after_sampling_tasks']])))
