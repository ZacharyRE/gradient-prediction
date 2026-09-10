"""Read-only compact progress for the frozen main and separate supplemental work."""
from datetime import datetime,timezone
from common import *
print(datetime.now(timezone.utc).isoformat(), 'hours_left', round((DEADLINE-time.time())/3600, 2))
pp=S/'results/confirmation_plan.json'
if not pp.exists():print('Main plan not yet frozen');raise SystemExit(0)
plan=json.loads(pp.read_text());main=list(json.loads((S/'results/confirmation_manifest.json').read_text()))
sup=json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text());extra=list(sup['models'].values())
for dataset in plan['datasets']:
    root=S/'results/evaluation'/dataset
    complete={role:[n for n in names if (root/n/'math_summary.json').exists()] for role,names in [('main',main),('supplement',extra)]}
    partial=[]
    for name in main+extra:
        p=root/name/'math_predictions.jsonl'
        if p.exists() and not (p.parent/'math_summary.json').exists():
            partial.append(dict(model=name,rows=len(p.read_text().splitlines())))
    print(dataset,json.dumps(dict(main_complete=len(complete['main']),supplement_complete=len(complete['supplement']),partial=partial)))
native=['base']+list(plan['families'][plan['primary_family']].values());complete=[];partial=[]
for name in native:
    root=S/'results/native/final_dev500'/name
    if (root/'summary.json').exists():complete.append(name)
    elif (root/'predictions.jsonl').exists():partial.append(dict(model=name,rows=len((root/'predictions.jsonl').read_text().splitlines())))
print('native',json.dumps(dict(complete=len(complete),partial=partial)))
extra_native=['gpu2_native_replay_base','gpu2_native_replay_student']
np=S/'audits/native_target_control_plan.json'
if np.exists():extra_native+=json.loads(np.read_text())['new_models']
extra_rows=[]
for name in extra_native:
    root=S/'results/native/dev'/name
    if (root/'summary.json').exists():
        v=json.loads((root/'summary.json').read_text());extra_rows.append(dict(model=name,complete=True,n=v['n'],correct=v['correct']))
    elif (root/'predictions.jsonl').exists():extra_rows.append(dict(model=name,complete=False,rows=len((root/'predictions.jsonl').read_text().splitlines())))
print('additional_native',json.dumps(extra_rows))
for fn in ['final_collection_status','native_target_control_status','expanded_unaveraged_supplement_status','additional_diagnostic_collection','final_campaign_completion','final_campaign_execution_reconciliation']:
    p=S/f'audits/{fn}.json'
    if p.exists():
        value=json.loads(p.read_text());print(fn,json.dumps({k:v for k,v in value.items() if k in ['status','all_complete','scored_datasets','analyses_complete','returncodes']}))
    else:print(fn,'waiting')
balanced=S/'audits/balanced_ood_execution.json'
if balanced.exists():
    b=json.loads(balanced.read_text())
    print('balanced_execution',json.dumps(dict(status=b['status'],main_gpu_complete=b['main_gpu_complete'],supplement_gpu_complete=b['supplement_gpu_complete'],parents={g:p['phase'] for g,p in b['parents'].items()},running=b['running'],tasks_complete=sum(t['status']=='complete' for t in b['tasks']),tasks_total=len(b['tasks']),errors=b['errors'])))
log=S/'logs/final_campaign_after_numeric_repair.log'
for line in log.read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    if r.get('event')=='child_finished' and r.get('returncode')!=0:
        planned=(r.get('name')=='benchmarks' and r['returncode']==-15 and (S/'audits/balanced_ood_gpu3_handoff.json').exists())
        print('PLANNED_CPU_HANDOFF_REQUIRES_FINAL_RECONCILIATION' if planned else 'ERROR',line)
