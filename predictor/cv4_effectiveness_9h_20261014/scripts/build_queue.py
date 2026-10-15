"""Small reusable queue builder; no processes launched when imported."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
PY='/mnt/micron/yixiaore/adapter/.venv/bin/python'
def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def stage(name,script,args):
    return dict(name=name,command=[PY,str(EXP/'scripts/stage.py'),'--name',name,PY,str(EXP/'scripts'/script),*map(str,args)])
def bundle(name,task,script,jobs,seed=101,pseed=123):
    config=EXP/'configs'/f'{name}_jobs.json';entries=EXP/'configs'/f'{name}_entries.json'
    write(config,jobs)
    args=['--task',task,'--jobs',config,'--seed',seed]
    if script not in ['online_ridge.py','online_ridge_v2.py','memory_cv.py']:args+=['--predictor-seed',pseed]
    return [stage(name,script,args),
        dict(name=name+'_entries',command=[PY,str(EXP/'scripts/make_entries.py'),'--task',task,'--jobs',str(config),'--seed',str(seed),'--predictor-seed',str(pseed),'--output',str(entries)]),
        stage(name+'_eval','evaluate.py',['--model','1p5b','--entries',entries,'--name',name,'--split','accuracy_dev','--max-new-tokens',256])]

if __name__=='__main__':
    fast=json.loads((EXP/'configs/fast_screen_paws.json').read_text())
    lr=json.loads((EXP/'configs/lr_screen_paws.json').read_text())
    anchor=json.loads((EXP/'configs/anchor_screen_paws.json').read_text())
    q=bundle('mnli_fast_v1','mnli','research_v3.py',fast)
    q+=bundle('mnli_lr_v1','mnli','research_v3.py',lr)
    q+=bundle('mnli_anchor_v1','mnli','anchored.py',anchor)
    write(EXP/'configs/queue_round4.json',q)
