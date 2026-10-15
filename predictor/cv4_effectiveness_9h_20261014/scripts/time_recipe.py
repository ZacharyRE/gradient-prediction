"""Standalone recipe timing without intermediate research diagnostics.

Includes model/data loading, pretrained predictor loading, prefix preparation,
updates and final checkpoint writing. Predictor pretraining and shared warmup
training are separate. Compare saved final weights with the evaluated reference.
"""
import time as _wall_time
SCRIPT_STARTED=_wall_time.perf_counter()
SCRIPT_STARTED_EPOCH=_wall_time.time()
from common import *
import argparse,importlib,sys

def assert_clean_gpu():
    holder=reservation_state();assert holder is not None
    output=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    foreign=[int(line.strip()) for line in output.splitlines() if int(line.strip()) not in [holder['pid'],os.getpid()]]
    assert not foreign,'Standalone benchmark requires clean occupancy; defer this replay until incidental contexts exit'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--recipe',required=True);ap.add_argument('--module',required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);ap.add_argument('--reference-adapter',required=True);ap.add_argument('--repeat',type=int,default=1);a=ap.parse_args()
    job=next(x for x in json.loads(Path(a.jobs).read_text()) if x['name']==a.recipe)
    replay_id=str(time.time_ns());name=f'timing__{a.recipe}__r{a.repeat}__{replay_id}'
    job=dict(job,name=name,steps=a.steps,save=[])
    path=EXP/'configs'/f'{a.task}_{name}.json';write(path,[job])
    assert_clean_gpu()
    os.environ['BENCH_FINAL_VALIDATION']='1'
    module=importlib.import_module(a.module)
    run_fn=getattr(module,{'online_ridge':'run_ridge','online_ridge_v2':'run_ridge','memory_cv':'run_memory'}.get(a.module,'run'))
    original_setup=run_fn.__globals__['setup'];captured={}
    def setup(*args,**kwargs):
        values=original_setup(*args,**kwargs);captured['model']=values[0];return values
    run_fn.__globals__['setup']=setup
    sys.argv=[a.module,'--task',a.task,'--jobs',str(path),'--seed',str(a.seed)]
    if a.module not in ['online_ridge','online_ridge_v2','memory_cv']:sys.argv+=['--predictor-seed',str(a.predictor_seed)]
    import torch
    started=time.perf_counter();module.main();model=captured['model']
    directory=EXP/'models'/f'{a.task}__{name}_u{a.seed}_p{a.predictor_seed}'
    if a.module in ['online_ridge','online_ridge_v2','memory_cv']:directory=EXP/'models'/f'{a.task}__{name}_u{a.seed}_p0'
    adapter=directory/'final-adapter';model.save_pretrained(adapter);torch.cuda.synchronize()
    total=time.perf_counter()-started
    script_total=time.perf_counter()-SCRIPT_STARTED
    finished_epoch=time.time()
    assert_clean_gpu()
    from safetensors.torch import load_file
    actual=load_file(str(adapter/'adapter_model.safetensors'));reference=load_file(str(Path(a.reference_adapter)/'adapter_model.safetensors'))
    assert actual.keys()==reference.keys()
    difference=max(float((actual[k]-reference[k]).abs().max()) for k in actual)
    relative=(sum(float((actual[k].double()-reference[k].double()).square().sum()) for k in actual)/max(1e-30,sum(float(reference[k].double().square().sum()) for k in actual)))**.5
    summary=json.loads((directory/'summary.json').read_text())
    history=json.loads((directory/'history.json').read_text())
    assert summary.get('complete') and history and history[-1]['step']==a.steps, 'Refuse to publish an incomplete timing replay'
    result=dict(task=a.task,recipe=a.recipe,module=a.module,seed=a.seed,predictor_seed=a.predictor_seed,steps=a.steps,repeat=a.repeat,standalone_pipeline_seconds=total,online_seconds=summary['online_seconds'],prefix_seconds=summary.get('cache_seconds',0),offline_initialization_seconds=summary.get('offline_fit_seconds',0),reference_adapter=a.reference_adapter,reference_sha256=sha(Path(a.reference_adapter)/'adapter_model.safetensors'),timed_adapter=str(adapter),timed_adapter_sha256=sha(adapter/'adapter_model.safetensors'),max_absolute_weight_difference=difference,relative_weight_difference=relative,weights_bitwise_equal=all(torch.equal(actual[k],reference[k]) for k in actual),timing_includes='Model/data/predictor loading, feature preparation, online updates, small history receipts, final adapter and any predictor checkpoint; excludes shared warmup training, inherited predictor pretraining and evaluation diagnostics',source_config_sha256=sha(a.jobs),harness_sha256=sha(Path(__file__)))
    result.update(standalone_script_seconds=script_total,script_timer_scope='From first script statement through final artifact save, including library imports and admission check. Excludes interpreter startup before script, shared warmup training, inherited offline predictor preparation, and post-run weight-comparison diagnostics.')
    result.update(timing_started_epoch=SCRIPT_STARTED_EPOCH,timing_finished_epoch=finished_epoch)
    write(EXP/'results/timing'/f'{a.task}_{a.recipe}_u{a.seed}_r{a.repeat}_{replay_id}.json',result);log(result)

if __name__=='__main__':main()
