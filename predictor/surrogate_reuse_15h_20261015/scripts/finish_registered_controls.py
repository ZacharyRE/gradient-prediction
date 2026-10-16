"""Finish short registered controls first, scoring completed models by phase."""
from common import *
import sys

def model_receipts_complete(item):
    args=item['training_argv'];arg=lambda k,d:args[args.index(k)+1] if k in args else d
    jobs=json.loads(Path(arg('--jobs','')).read_text())
    for job in jobs:
        root=EXP/'models'/f'mnli__w301_{arg("--subset","update")}_{job["name"]}_u{arg("--seed","401")}{arg("--suffix","")}'
        if not (root/'summary.json').exists():return False
        summary=json.loads((root/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text())
        if not summary.get('complete') or cfg['job']!=job:return False
    return True

def main():
    policy=json.loads((EXP/'configs/registered_controls_completion_policy.json').read_text())
    ready=EXP/'results/kd_readout_completion.json'
    while not (ready.exists() and json.loads(ready.read_text()).get('complete')):
        if time.time()>=policy['training_cutoff_epoch']:
            write(EXP/'results/registered_controls_completion.json',dict(finished=True,executed=[],reason='KD readout consumed the remaining optional training window'));return
        time.sleep(10)
    phases=[[n for n in policy['stages'] if n.startswith('needed_')],['pilot_class_batched'],['pilot_token'],['pilot_table']]
    executed=[]
    def stage(name,command):
        tick=time.time();code=subprocess.call([sys.executable,str(EXP/'scripts/stage.py'),'--name','finish_controls_'+name,*command],cwd=ROOT)
        executed.append(dict(stage=name,returncode=code,started=tick,finished=time.time()))
        write(EXP/'results/registered_controls_completion.json',dict(finished=False,executed=executed))
    for phase in phases:
        for name in phase:
            if time.time()>=policy['training_cutoff_epoch']:break
            if model_receipts_complete(policy['stages'][name]):continue
            stage(name,[sys.executable,str(EXP/'scripts/complete_registered_controls.py'),name])
        subprocess.run([sys.executable,str(EXP/'scripts/collect_optional_readouts.py')],check=True,cwd=ROOT)
        inventory=json.loads((EXP/'results/optional_completed_readout_inventory.json').read_text())
        if inventory['unscored_completed_entries'] and time.time()<policy['readout_cutoff_epoch']:
            stage('eval_'+inventory['evaluation_name'],[sys.executable,str(EXP/'scripts/evaluate.py'),'--model','1p5b','--entries',inventory['entries_path'],'--name',inventory['evaluation_name'],'--split','accuracy_dev','--max-new-tokens','256'])
        if time.time()>=policy['training_cutoff_epoch']:break
    subprocess.run([sys.executable,str(EXP/'scripts/collect_optional_readouts.py')],check=True,cwd=ROOT)
    inventory=json.loads((EXP/'results/optional_completed_readout_inventory.json').read_text())
    write(EXP/'results/registered_controls_completion.json',dict(finished=True,epoch=time.time(),executed=executed,
          all_registered_recipes_completed=not inventory['uncompleted_recipes'],
          all_completed_recipes_scored=inventory['unscored_completed_entries']==0,
          scope='Unchanged registered development controls; short controls precede long memory-table grids. Fully completed recipes are scored by phase. Partial/unexecuted recipes are not measured failures.'))

if __name__=='__main__':main()
