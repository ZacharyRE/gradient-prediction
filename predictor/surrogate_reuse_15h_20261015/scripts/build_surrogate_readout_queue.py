"""Materialize final readouts only after the recipe/adapter freeze exists."""
from common import *
from surrogate_freeze import FREEZE,DECISIONS,critical_code
import sys

def main():
    frozen=json.loads(FREEZE.read_text());assert frozen['complete'] and frozen['code']==critical_code()
    entries=[frozen['entries'][key]['entry'] for key in sorted(frozen['entries'])]
    path=EXP/'configs/surrogate_final_entries.json'
    if path.exists():assert json.loads(path.read_text())==entries
    else:write(path,entries)
    queue=[];models=sorted(frozen.get('behavior_predictors',{}))
    def add(name,script,args=(),gpu=False):
        command=[sys.executable,str(EXP/'scripts'/script),*map(str,args)]
        if gpu:command=[sys.executable,str(EXP/'scripts/stage.py'),'--name',name,*command]
        queue.append(dict(name=name,command=command))
    for split in ['selection_dev','selection_holdout']:
        name='surrogate_final_'+split
        add(name,'evaluate.py',['--model','1p5b','--entries',path,'--name',name,'--split',split,'--max-new-tokens',256],True)
        if models:
            name='surrogate_final_behavior_'+split
            add(name,'evaluate_behavior.py',['--entries',path,'--predictors',*models,'--name',name,'--split',split,'--teacher'],True)
        if split=='selection_dev':add('surrogate_final_freeze_choices','freeze_selection_decisions.py')
    add('surrogate_final_analysis','analyze_selection.py',['--analysis-spec',EXP/'configs/surrogate_analysis_spec.json'])
    add('surrogate_final_correctness_bundle','export_prediction_bundle.py')
    target=EXP/'configs/queue_surrogate_final_readouts.json'
    if target.exists():assert json.loads(target.read_text())==queue
    else:write(target,queue)
    write(EXP/'results/surrogate_readout_queue_receipt.json',dict(freeze_sha256=sha(FREEZE),entries_sha256=sha(path),queue_sha256=sha(target),entries=len(entries),behavior_models=models,scope='No recipe selection. All generated and behavior dev readouts precede fixed candidate choices; all holdout readouts follow the choice freeze. Execute with BENCH_FINAL_VALIDATION=1 under the reserved GPU stage runner.'))
    log(dict(entries=len(entries),stages=len(queue),queue=str(target)))

if __name__=='__main__':main()
