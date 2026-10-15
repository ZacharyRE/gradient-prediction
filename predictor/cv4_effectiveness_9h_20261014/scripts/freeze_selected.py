"""Review/freeze an explicit development-selected cohort plan before final readouts.

Plan: {comparison_reference: {task: cohort}, selection_reason: str,
       cohorts: [{task, cohort, tags: [...], primary_candidate: bool}]}.
Without --write, print the exact review table without changing the freeze.
"""
from common import *
import argparse,csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--write',action='store_true');a=ap.parse_args()
    plan=json.loads(a.plan.read_text());assert plan['selection_reason']
    development={r['tag']:r for r in csv.DictReader((EXP/'results/development_live.csv').open()) if r['dev_accuracy_pct']}
    artifacts={}
    for path in (EXP/'models').glob('*/summary.json'):
        for e in json.loads(path.read_text()).get('entries',[]):
            assert e['tag'] not in artifacts
            artifacts[e['tag']]=e
    entries=[];table=[];cohort_names=set()
    for cohort in plan['cohorts']:
        task=cohort['task'];name=cohort['cohort'];assert (task,name) not in cohort_names;cohort_names.add((task,name))
        selected=[]
        for tag in cohort['tags']:
            assert tag in development and tag in artifacts,tag
            e=dict(artifacts[tag]);assert e['task']==task
            assert sha(Path(e['adapter'])/'adapter_model.safetensors')==e['adapter_sha256']
            e.update(cohort=name,primary_candidate=bool(cohort.get('primary_candidate',False)))
            entries.append(e);selected.append(e)
        assert sorted(e['seed'] for e in selected)==[101,102,103],(task,name,'Require the three paired update seeds')
        assert len({e['steps'] for e in selected})==1 and len({e['lr'] for e in selected})==1
        assert len({e['method'] for e in selected})==1,'Do not choose different methods for different seeds'
        settings=[]
        for e in selected:
            cfg=json.loads((Path(e['adapter']).parent/'config.json').read_text())
            settings.append(json.dumps({k:v for k,v in cfg['job'].items() if k not in ['name','steps','save','predictor_path']},sort_keys=True))
        assert len(set(settings))==1,'Algorithm settings must agree across paired seeds'
        table.append(dict(task=task,cohort=name,method_names=sorted({e['method'] for e in selected}),steps=selected[0]['steps'],lr=selected[0]['lr'],dev_mean_pct=float(np.mean([float(development[e['tag']]['dev_accuracy_pct']) for e in selected])),dev_seed_values_pct=[float(development[e['tag']]['dev_accuracy_pct']) for e in selected],online_seconds_mean=float(np.mean([e['online_seconds'] for e in selected])),prefix_seconds_mean=float(np.mean([e.get('cache_seconds',0) for e in selected])),primary_candidate=cohort.get('primary_candidate',False)))
        table[-1]['predictor_seeds']=sorted({e.get('predictor_seed',0) for e in selected})
    tasks=sorted({e['task'] for e in entries})
    for task in tasks:
        assert (task,plan['comparison_reference'][task]) in cohort_names
        assert sum(r['primary_candidate'] for r in table if r['task']==task)==1,'Declare exactly one primary predictor candidate per task'
        warm=dict(json.loads((EXP/'configs'/f'{task}_warmup.json').read_text()))
        assert sha(Path(warm['adapter'])/'adapter_model.safetensors')==warm['adapter_sha256']
        warm.update(cohort='warmup',primary_candidate=False,online_seconds=0.,cache_seconds=0.)
        entries.append(warm)
    assert len({e['tag'] for e in entries})==len(entries),'An adapter should belong to one cohort in this manifest'
    output=dict(frozen_epoch=time.time(),selection_split='accuracy_dev',selection_reason=plan['selection_reason'],selection_plan_sha256=sha(a.plan),comparison_reference=plan['comparison_reference'],entries=entries,development_review=table,evaluation_names={task:{'oracle_confirmation':f'{task}_frozen_confirmation','test':f'{task}_frozen_test'} for task in tasks},selection_data_sha256={task:sha(data(task,'accuracy_dev')) for task in tasks},quality_margin_pp=1.,scope='Three paired update seeds, conditional on fixed task warmups; exploratory test populations previously used in older studies. No changes to selected recipes after frozen confirmation/test readouts.')
    log(dict(write=a.write,review=table,total_adapters=len(entries)))
    if a.write:
        target=EXP/'configs/final_freeze.json';assert not target.exists(),'Freeze is immutable; preserve the existing selection'
        for protocol in (EXP/'results/eval').glob('*/protocol.json'):
            old=json.loads(protocol.read_text())
            assert old.get('args',{}).get('split') not in ['oracle_confirmation','test'],'Final readout already exists; do not create a retrospective freeze'
        write(target,output)
        for task in tasks:write(EXP/'configs'/f'{task}_frozen_entries.json',[e for e in entries if e['task']==task])


if __name__=='__main__':main()
