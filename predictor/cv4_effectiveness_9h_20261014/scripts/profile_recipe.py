"""Replay a frozen recipe with native matrix-FLOP counting; not a speed benchmark.

Counts include prefix/feature preparation and online updates, but not fitting an
already-existing predictor. Scalar/nonlinear/normalization/optimizer arithmetic
is excluded. Full sequences and native fused predictor operators are retained.
"""
from common import *
from flop_accounting import NativeFlopCounterMode,FUSED_MAPPING,OperatorCoverage,coverage_record
import argparse,importlib,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--recipe',required=True);ap.add_argument('--module',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);ap.add_argument('--steps',type=int);a=ap.parse_args()
    job=next(x for x in json.loads(Path(a.jobs).read_text()) if x['name']==a.recipe)
    replay_id=str(time.time_ns())
    job=dict(job,name='profile__'+job['name']+'__'+replay_id,save=[])
    if a.steps is not None:
        assert a.steps>0
        job['steps']=a.steps
    path=EXP/'configs'/f'profile_{a.task}_{a.recipe}_u{a.seed}_{replay_id}.json';write(path,[job])
    # Preserve the numerical trainer; only checkpoint diagnostics are disabled.
    os.environ['BENCH_FINAL_VALIDATION']='1'
    module=importlib.import_module(a.module)
    argv=[a.module,'--task',a.task,'--jobs',str(path),'--seed',str(a.seed)]
    if a.module not in ['online_ridge','online_ridge_v2','memory_cv']:argv+=['--predictor-seed',str(a.predictor_seed)]
    sys.argv=argv;counter=NativeFlopCounterMode(display=False,custom_mapping=FUSED_MAPPING)
    coverage=OperatorCoverage();steps=[]
    function_name={'anchored':'anchored_update','anchored_fastio':'anchored_update','stratified':'stratified_update','residual_ema':'update_smoothed','time_query':'queried_update'}.get(a.module,'update')
    if hasattr(module,function_name):
        original=getattr(module,function_name)
        def wrapped(*args,**kwargs):
            before=counter.get_total_flops();row=original(*args,**kwargs)
            steps.append(dict(step=row['step'],matrix_flops=counter.get_total_flops()-before))
            return row
        setattr(module,function_name,wrapped)
    tick=time.time()
    with counter,coverage:module.main()
    actual_pseed=0 if a.module in ['online_ridge','online_ridge_v2','memory_cv'] else a.predictor_seed
    run_directory=EXP/'models'/f'{a.task}__{job["name"]}_u{a.seed}_p{actual_pseed}'
    summary=json.loads((run_directory/'summary.json').read_text())
    history=json.loads((run_directory/'history.json').read_text())
    assert summary.get('complete') and history and history[-1]['step']==job['steps'], 'Refuse to publish an incomplete or deadline-shortened FLOP replay'
    if steps:assert len(steps)==job['steps'],'Every requested update must be counted'
    result=dict(task=a.task,recipe=a.recipe,module=a.module,steps=job['steps'],seed=a.seed,predictor_seed=a.predictor_seed,total_matrix_flops=counter.get_total_flops(),online_matrix_flops=sum(r['matrix_flops'] for r in steps) if steps else None,step_counts=steps,coverage=coverage_record(coverage,counter),instrumented_wall_seconds=time.time()-tick,source_jobs_sha256=sha(a.jobs),profiler_sha256=sha(Path(__file__)),counter_sha256=sha(EXP/'scripts/flop_accounting.py'),scope='Preparation plus online update matrix arithmetic. Excludes inherited offline fitting/gradient collection, model loading I/O, scalar/normalization/nonlinear/optimizer arithmetic; instrumented timings are not speed measurements.')
    write(EXP/'results/flops'/f'{a.task}_{a.recipe}_u{a.seed}_steps{job["steps"]}.json',result)
    log({k:v for k,v in result.items() if k not in ['coverage','step_counts']})

if __name__=='__main__':main()
