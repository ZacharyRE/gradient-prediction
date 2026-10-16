"""Materialize the frozen readout/replay schedule; never select from readouts.

Each final_selection_plan cohort declares replay_module and optional profile.
Run the generated queue with BENCH_FINAL_VALIDATION=1 and the study GPU env.
"""
from common import *
import sys


def main():
    freeze_path=EXP/'configs/final_freeze.json'
    freeze=json.loads(freeze_path.read_text())
    plan=json.loads((EXP/'configs/final_selection_plan.json').read_text())
    policy=json.loads((EXP/'configs/final_timing_policy.json').read_text())
    target=EXP/'configs/final_timing_plan.json'
    assert not target.exists(),'Preserve the registered final replay schedule'
    specs={(r['task'],r['cohort']):r for r in plan['cohorts']}
    scripts=EXP/'scripts';queue=[];timing=[]
    def stage(name,script,args):
        queue.append(dict(name=name,command=[sys.executable,str(scripts/'stage.py'),'--name',name,sys.executable,str(scripts/script),*map(str,args)]))
    for split in ['oracle_confirmation','test']:
        for task in sorted(freeze['evaluation_names']):
            name=freeze['evaluation_names'][task][split]
            stage(name,'evaluate.py',['--model','1p5b','--entries',EXP/'configs'/f'{task}_frozen_entries.json','--name',name,'--split',split,'--max-new-tokens',256])
    # Complete CPU bootstrap work before standalone timing, avoiding our own
    # analysis workload competing with benchmark CPU execution.
    for split in ['oracle_confirmation','test']:
        queue.append(dict(name=f'frozen_{split}_analysis',command=[sys.executable,str(scripts/'analyze_frozen_results.py'),'--split',split,'--bootstrap','5000']))
    entries=[e for e in freeze['entries'] if e['cohort']!='warmup']
    for round_index,round_spec in enumerate(policy['rounds']):
        for task in sorted(freeze['evaluation_names']):
            selected=sorted([e for e in entries if e['task']==task and e['seed']==round_spec['update_seed']],key=lambda e:e['cohort'])
            offset=round_index%len(selected);selected=selected[offset:]+selected[:offset]
            for e in selected:
                spec=specs[task,e['cohort']];module=spec['replay_module']
                cfg=json.loads((Path(e['adapter']).parent/'config.json').read_text())
                job=cfg['job'];assert job['name']==e['method']
                jobs=EXP/'configs'/f'frozen_{task}_{e["cohort"]}_u{e["seed"]}_jobs.json'
                write(jobs,[job]);name=f'final_timing_{task}_{e["cohort"]}_r{round_spec["repeat"]}'
                args=['--task',task,'--jobs',jobs,'--recipe',e['method'],'--module',module,'--steps',e['steps'],'--seed',e['seed'],'--predictor-seed',e['predictor_seed'],'--reference-adapter',e['adapter'],'--repeat',round_spec['repeat']]
                stage(name,'time_recipe.py',args)
                timing.append(dict(task=task,cohort=e['cohort'],seed=e['seed'],predictor_seed=e['predictor_seed'],repeat=round_spec['repeat'],reference_sha256=e['adapter_sha256'],module=module,command=queue[-1]['command']))
    for e in entries:
        spec=specs[e['task'],e['cohort']]
        if e['seed']!=101 or not spec.get('profile',False):continue
        jobs=EXP/'configs'/f'frozen_{e["task"]}_{e["cohort"]}_u101_jobs.json'
        stage(f'final_flops_{e["task"]}_{e["cohort"]}','profile_recipe.py',['--task',e['task'],'--jobs',jobs,'--recipe',e['method'],'--module',spec['replay_module'],'--steps',e['steps'],'--seed',101,'--predictor-seed',e['predictor_seed']])
    write(target,dict(created_epoch=time.time(),freeze_sha256=sha(freeze_path),timing_policy_sha256=sha(EXP/'configs/final_timing_policy.json'),rows=timing))
    write(EXP/'configs/queue_final.json',queue)
    log(dict(stages=len(queue),timing_replays=len(timing),freeze_sha256=sha(freeze_path)))


if __name__=='__main__':main()
